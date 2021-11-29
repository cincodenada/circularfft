extern crate piston_window;
extern crate tuple;

mod bracketed_chunks;
mod spectrogram;

use bracketed_chunks::*;
use spectrogram as spec;

use piston_window::*;
use tuple::*;

use druid::{AppLauncher, WindowDesc, PlatformError};
use druid::{Data, Lens};
use druid::{Color, RenderContext};
use druid::{Command, Selector};
use druid::{Widget, WidgetExt, Env, WidgetId, LifeCycle, Event};
use druid_widget_nursery::DropdownSelect;
use druid::widget::{Controller, Flex, Label, Painter, IdentityWrapper};
use druid::piet::kurbo::{Line, BezPath, PathSeg, PathEl};
use druid::im;

const FFT_CALC_SELECTOR: Selector<()> = Selector::new("fft_calc");

use std::env;
use std::time::{Duration, SystemTime};
use std::fs::File;
use std::path::Path;
use wav::BitDepth;
//use inline_python::python;
use ordered_float::OrderedFloat;
use itertools::Itertools;

struct OctaveSharder {
    min: spec::Freq,
    max: spec::Freq
}
impl Sharder<&spec::Bin> for OctaveSharder {
    fn shard(&self, freq: &&spec::Bin) -> Option<usize> {
        match *freq {
            v if v.freq < self.min => None,
            v if v.freq > self.max => None,
            v => Some(v.freq_whole as usize)
        }
    }
}

type ScreenPoint = (f64, f64);
type FreqRange = (f32, f32);
type OctRange = (usize, usize);

struct Colorer<T> {
    min: T,
    max: T,
    halfrange: T
}
impl Colorer<spec::Freq> {
    fn new(min: spec::Freq, max: spec::Freq) -> Self {
        let min = min.log2();
        let max = max.log2();
        let halfrange = (max - min)/2.0;
        Colorer { min, max, halfrange }
    }

    fn map(&self, val: spec::Freq) -> ArrayColor {
        match (val.log2() - self.min)/self.halfrange {
            v if v > 0.0 && v <= 1.0 => [0.0, v/2.0, 0.0, 1.0].into(),
            v => [v-1.0, v/2.0, 0.0, 1.0].into(),
        }
    }
}

struct ArrayColor {
    color: [f32;4]
}
impl From<[f32;4]> for ArrayColor {
    fn from(c: [f32;4]) -> ArrayColor { ArrayColor { color: c } }
}
impl Into<[f32;4]> for ArrayColor {
    fn into(self) -> [f32;4] { self.color }
}
//impl From<ArrayColor> for Color {
//    fn from(c: ArrayColor) -> Color { c.into() }
//}
impl Into<Color> for ArrayColor {
    fn into(self) -> Color {
        let c = self.color.map(|v| v as f64);
        Color::rgba(c[0], c[1], c[2], c[3])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let fftpow = args[1].parse::<u32>().unwrap_or(13);
    let overlap = args[2].parse::<f32>().unwrap_or(0.5);
    let speed = args[3].parse::<f32>().unwrap_or(1.0);

    let fftsize = 2_usize.pow(fftpow);
    println!("Using fft size of {}, overlap {}", fftsize, overlap);

    let mut inp_file = File::open(Path::new("input.wav"))?;
    let (header, data) = wav::read(&mut inp_file)?;

    let samples : Vec<_> = match data {
        //BitDepth::Sixteen(vec) => vec.into_iter().collect(),
        //BitDepth::TwentyFour(vec) => vec.into_iter().collect(),
        //BitDepth::ThirtyTwoFloat(vec) => vec.into_iter().collect(),
        // TODO: We probably shouldn't need to collect() here
        BitDepth::ThirtyTwoFloat(vec) => vec,
        //BitDepth::Empty => panic!("Ack!")
        _ => panic!("Ack!"),
    };
    dbg!(&header);

    //let floatMax = |a:f32, b:f32| max(OrderedFloat(a), OrderedFloat(b)).into();

    let mut spectrogram = spec::Spectrogram::from_samples(
        &samples, header.sampling_rate, header.channel_count
    );
    spectrogram.calculate_with(spec::Params {
        fft_size: fftsize,
        overlap,
        window_type: spec::Window::Hann
    });
    let ms_per_col = (((fftsize as f32 * (1.0-overlap))/header.sampling_rate as f32)/speed*1000.0) as u64;
    dbg!(ms_per_col);

    make_druid_window(spectrogram);
    //make_window(&spectrogram, colorer, Duration::from_millis(ms_per_col.into()));
    //make_circle_plot(&spectrogram);
    //make_rect_plot(&spectrogram);
    Ok(())
}

#[derive(Clone, Data)]
struct Counter(i32);



#[derive(Data, Clone, Lens)]
struct AppData {
    fft_size: usize,
    window_type: spec::Window,
    overlap: f32,
    // TODO: Updating depends on manually requesting paint, that's icky, data-fy this
    #[data(ignore)]
    fft_cols: im::Vector<spec::Column>,
    #[data(ignore)]
    val_range: (f32, f32),
}

struct FftParameter {
    fft_widget_id: WidgetId
}
impl<W: Widget<AppData>> Controller<AppData, W> for FftParameter {
    fn update(&mut self, child: &mut W, ctx: &mut druid::UpdateCtx, old_data: &AppData, data: &AppData, env: &Env) {
        child.update(ctx, old_data, data, env);
        if !data.same(old_data) {
            dbg!("Requesting recalculation");
            ctx.submit_command(Command::new(FFT_CALC_SELECTOR, (), self.fft_widget_id));
        }
    }
}
struct FftWidget { spectrogram: spec::Spectrogram }
impl FftWidget {
    fn recalculate(&mut self, data: &mut AppData) {
        dbg!("Recalculating");
        self.spectrogram.calculate_with(spec::Params {
            fft_size: data.fft_size,
            overlap: data.overlap,
            window_type: data.window_type
        });
        data.val_range = (self.spectrogram.min_mag, self.spectrogram.max_mag);
        // TODO: This clone shouldn't be necessary, from() should do a clone I think
        data.fft_cols = self.spectrogram.columns.clone().into();
    }
}
impl<W: Widget<AppData>> Controller<AppData, W> for FftWidget {
    fn lifecycle(&mut self, child: &mut W, ctx: &mut druid::LifeCycleCtx, event: &druid::LifeCycle, data: &AppData, env: &Env) {
        child.lifecycle(ctx, event, data, env);
        match event {
            LifeCycle::WidgetAdded => { ctx.submit_command(Command::new(FFT_CALC_SELECTOR, (), ctx.widget_id())); }
            _ => {}
        }
    }
    fn event(&mut self, child: &mut W, ctx: &mut druid::EventCtx, event: &druid::Event, data: &mut AppData, env: &Env) {
        match event {
            Event::Command(cmd) if cmd.is(FFT_CALC_SELECTOR) => {
                self.recalculate(data);
                ctx.request_paint();
            },
            _ => child.event(ctx, event, data, env)
        }
    }

}


fn make_druid_window(spectrogram: spec::Spectrogram) -> Result<(), PlatformError> {
    let state = AppData {
        fft_size: spectrogram.params.fft_size,
        window_type: spectrogram.params.window_type,
        overlap: spectrogram.params.overlap,
        fft_cols: im::Vector::new(),
        val_range: (0.0, 0.0)
    };

    // Window builder. We set title and size
    let main_window = WindowDesc::new(build_druid_window(spectrogram))
        .title("Spectrogram Toy")
        .window_size((200.0, 100.0));

    // Run the app
    AppLauncher::with_window(main_window)
        .launch(state)
}

fn build_druid_window(spectrogram: spec::Spectrogram) -> impl Widget<AppData> {
    let fft_widget_id = WidgetId::next();

    let fft_size = Flex::row()
        .with_child(Label::new("Fft Size:").align_left())
        .with_child(DropdownSelect::<usize>::new(vec![
                ("4096", 4096),
                ("8192", 8192),
                ("16384", 16384)
            ])
            .align_left()
            .lens(AppData::fft_size)
            .controller(FftParameter{fft_widget_id})
        );
    let window_type = Flex::row()
        .with_child(Label::new("Window Type:").align_left())
        .with_child(DropdownSelect::<spec::Window>::new(vec![
                ("Square", spec::Window::Square),
                ("Hann", spec::Window::Hann),
                ("Hamming", spec::Window::Hamming),
                ("Blackman", spec::Window::Blackman)
            ])
            .align_left()
            .lens(AppData::window_type)
            .controller(FftParameter{fft_widget_id})
        );

    // Container for the whole UI
    let controls = Flex::column()
        .with_child(fft_size)
        .with_child(window_type);

    let freq_range = (16.35, 8372.02);
    let mapped_range = freq_range.map(|v| (v as f64).log2());
    let mapped_span = mapped_range.1 - mapped_range.0;

    let fft = IdentityWrapper::wrap(Painter::new(move |ctx, data: &AppData, _env| {
        dbg!(data.fft_cols.len());
        if data.fft_cols.len() < 50 { return }

        // TODO: Push colorer itself into state?
        let colorer = Colorer::new(data.val_range.0, data.val_range.1);

        //ctx.clear([0.5, 0.5, 0.5, 1.0]);
        //let col = &data.fft_cols[50];
        //let rects = make_wedges(octave_bins(&col, freq_range), freq_range.0);
        let rects = data.fft_cols.iter().enumerate().map(|(idx, col)| {
            col.bins.iter().map(move |bin| (SlimBin::from_bin(bin, (idx as f64, (idx+1) as f64))))
        }).flatten();
        let dims = ctx.size();

        let min_dim = std::cmp::min_by(dims.width, dims.height, |a, b| a.partial_cmp(b).unwrap());
        let xscale = dims.width/data.fft_cols.len() as f64;
        let first_bins = &data.fft_cols[0].bins;
        let yscale = dims.height/first_bins[first_bins.len() - 1].freq.log2() as f64;
        // TODO: Make this a viewport
        rects.map(|bin| {
            let rect = druid::Rect::new(
                bin.xrange.0*xscale, bin.yrange.0.log2()*yscale,
                bin.xrange.1*xscale, bin.yrange.1.log2()*yscale
            );
            let color: Color = colorer.map(bin.val).into();
            ctx.fill(rect, &color)
        }).last();
    }), fft_widget_id).controller(FftWidget{ spectrogram });

    Flex::row()
        .with_flex_child(fft.expand(), 1.0)
        .with_child(controls)
}

fn make_window(spectrogram: &spec::Spectrogram, colorer: Colorer<spec::Freq>, spf: Duration) {
    let freq_range = (16.35, 8372.02);
    let mapped_range = freq_range.map(|v| (v as f64).log2());
    let mapped_span = mapped_range.1 - mapped_range.0;

    let mut slice = spectrogram.columns.iter().cycle();
    let mut col = slice.next().unwrap();
    let mut window: PistonWindow =
        WindowSettings::new("Hello World!", [512; 2])
            .build().unwrap();
    let mut time = SystemTime::now();
    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g, _| {
            if let Ok(v) = time.elapsed() {
                if v > spf {
                    col = slice.next().unwrap();
                    time = SystemTime::now();
                }
            }

            clear([0.5, 0.5, 0.5, 1.0], g);
            let rects = make_wedges(octave_bins(col, freq_range), freq_range.0);
            let dims = c.viewport.unwrap().draw_size.map(f64::from);
            rects.into_iter().map(|(val, points)| polygon(
                colorer.map(val).into(),
                &points.map(|p| [
                    (dims[0]/2.0)*(p[0]/mapped_span+1.0),
                    (dims[1]/2.0)*(p[1]/mapped_span+1.0),
                ]),
                c.transform, g
            )).last();
        });
    }
}

fn make_circle_plot(spectrogram: &spec::Spectrogram) {
    let freqbins: Vec<f32> = (1..spectrogram.half_size).map(|v| (v as f32).log2()).collect::<Vec<_>>();
    
    let r: Vec<f32> = freqbins.iter().map(|v| v.floor()).collect();
    //let theta = r.iter().zip(freqbins.iter()).map(|(&r, &v)| v - r).collect::<Vec<_>>();
    let xbins = r.to_vec().into_iter().map(OrderedFloat).max().unwrap();
    let xbinsf: f32 = xbins.into();

    let onefreq = freqbins.iter().filter(|f| OrderedFloat(**f) >= xbins).map(|f| f - xbinsf).collect::<Vec<_>>();

    //dbg!(&theta);
    //dbg!(&r);
    //dbg!(&xbinsf);
    //dbg!(&onefreq);
    //dbg!(&freqbins);
    //dbg!(&mag[200]);
    //dbg!(&dupcol);
    //dbg!(&wholes);

    let mag = spectrogram.columns[100].bins.iter().map(|b| b.mag.log2()).collect::<Vec<_>>();
    let (_x, _y, _values) = make_color_mesh(&mag, &freqbins, &onefreq, xbinsf);

    //python! {
    //    import matplotlib.pyplot as plt
    //    import numpy as np
    //    import math

    //    x = [row + [math.pi*2] for row in 'x]
    //    x = x + [x[-1]]
    //    y = [col + [col[-1]] for col in 'y] + [['y[-1][0]+1] * (len('y[0])+1)]
    //    def dims(x):
    //        print(len(x))
    //        print([len(r) for r in x])
    //    dims(x)
    //    dims(y)
    //    dims('values)

    //    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    //    ax.set_rmax(3)
    //    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    //    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    //    ax.set_xticks([(s+0.5)/12*math.pi*2 for s in range(0,12)])
    //    ax.set_xticklabels(['|']*12)
    //    ax.grid(True)

    //    plt.pcolormesh(x, y, 'values)
    //    plt.show()
    //}
}

fn make_rect_plot(spectrogram: &spec::Spectrogram) {
    //let starts: Vec<usize> = (0..width).map(|v| v*fftsize/2).collect();
    //let time: Vec<usize> = starts.iter().map(|start| vec![*start+fftsize/4;fftsize/2]).flatten().collect();
    //let freqbins: Vec<f32> = (0..fftsize/2).map(|v| floatMax((v as f32).log10(),0.0)).collect::<Vec<_>>();
    //let freq: Vec<f32> = starts.iter().map(|_| freqbins.to_vec()).flatten().collect();

    let cols = spectrogram.columns.iter().enumerate();
    let vals = cols.clone().skip(1).map(|(idx, c)| c.bins.iter().skip(1)
        .map(move |b| (idx*spectrogram.half_size, b.freq.log2(), b.mag.log2()))
    ).flatten();
    let time = vals.clone().map(|v| v.0).collect::<Vec<_>>();
    let freq = vals.clone().map(|v| v.1).collect::<Vec<_>>();
    let mag = vals.clone().map(|v| v.2).collect::<Vec<_>>();
    dbg!(&time.len());
    dbg!(&freq.len());
    dbg!(&mag.len());

    let bincount = spectrogram.columns.len()-1;
    let freqbins = spectrogram.columns[0].bins.iter().skip(1).map(|b| b.freq.log2()).collect::<Vec<_>>();
    dbg!(&bincount);
    dbg!(&freqbins.len());

    //python! {
    //    import matplotlib.pyplot as plt

    //    plt.hist2d('time, 'freq, ['bincount, 'freqbins],weights='mag)
    //    plt.show()
    //}

}


fn make_color_mesh(fftcol: &[f32], freqbins: &[f32], onefreq: &[f32], repcount: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut col = fftcol.iter();
    let mut freqiter = freqbins.windows(2);
    let mut curcol = col.next().unwrap();
    let mut curfreq = freqiter.next();
    let wholes: Vec<Vec<_>> = (0..=repcount as usize).map(|v| std::iter::repeat(v as f32).take(onefreq.len()).collect()).collect();
    let dupcol: (Vec<Vec<_>>, Vec<Vec<_>>) = (0..=repcount as usize).map(|whole| onefreq.iter().map(|frac| {
        let comp = *frac + whole as f32;
        match curfreq {
            Some([_min, max]) if comp >= *max => {
                curfreq = freqiter.next();
                curcol = col.next().unwrap();
            },
            _ => {}
        }
        (*frac * std::f32::consts::PI*2.0, *curcol)
    }).unzip()).unzip();

    (dupcol.0, wholes, dupcol.1)
}

fn dbg_iter<I, T>(it: I) -> impl Iterator<Item=T> where I: Iterator<Item=T>, T: std::fmt::Debug {
    let collected = it.collect::<Vec<_>>();
    dbg!(&collected);
    collected.into_iter()
}

struct SlimBin {
    xrange: ScreenPoint,
    yrange: ScreenPoint,
    val: spec::Mag
}
impl SlimBin {
    fn from_bin(bin: &spec::Bin, time_range: ScreenPoint) -> SlimBin {
        SlimBin {
            val: bin.mag,
            xrange: bin.freq_range.map(|v| v as f64),
            yrange: time_range.map(|v| v as f64)
        }
    }
}
impl From<(spec::Mag, FreqRange, OctRange)> for SlimBin {
    fn from((mag, freq_range, time_range): (spec::Mag, FreqRange, OctRange)) -> Self {
        SlimBin {
            val: mag,
            xrange: freq_range.map(|v| v as f64),
            yrange: time_range.map(|v| v as f64)
        }
    }
}

fn octave_bins<'a>(col: &'a spec::Column, clip: FreqRange) -> impl Iterator<Item=SlimBin> + 'a {
    let sharder = OctaveSharder { min: clip.0, max: clip.1 };
    let bounds = (0.0, 1.0);
    col.bins.iter().skip(1)
        .filter(move |bin| bin.freq >= clip.0 && bin.freq < clip.1)
        .bracketed_chunks(sharder)
        .tuple_windows()
        .map(move |v| {
            match v {
                (ShardResult::Start(cur_shard, cur), ShardResult::Start(next_shard, _)) => vec![
                    (
                        cur.mag,
                        (bounds.0, cur.freq_fract),
                        (cur_shard, next_shard)
                    ),
                    (
                        cur.mag,
                        (cur.freq_fract, bounds.1),
                        (cur_shard, next_shard)
                    )
                ],
                (ShardResult::Start(_, prev), ShardResult::Item(cur_shard, cur)) => vec![
                    (
                        cur.mag,
                        (bounds.0, prev.freq_fract),
                        (cur_shard, cur_shard + 1)
                    ),
                    (
                        cur.mag,
                        (prev.freq_fract, cur.freq_fract),
                        (cur_shard, cur_shard + 1)
                    )
                ],
                (ShardResult::Item(_, prev), ShardResult::Item(cur_shard, cur)) => vec![
                    (
                        cur.mag,
                        (prev.freq_fract, cur.freq_fract),
                        (cur_shard, cur_shard + 1)
                    )
                ],
                (ShardResult::Item(_, prev), ShardResult::Start(cur_shard, cur)) => vec![
                    (
                        cur.mag,
                        (prev.freq_fract, bounds.1),
                        (cur_shard, cur_shard + 1)
                    )
                ],
                _ => vec![]
            }
        }).flatten().map(|v| v.into())
}

fn make_wedges<'a>(bins: impl Iterator<Item=SlimBin> + 'a, ymin: f32) -> impl Iterator<Item=(f32, [[f64;2];4])> + 'a {
    let polar = move |thetaish, r, min: f32| {
        let theta: f64 = thetaish*2.0*std::f64::consts::PI;
        let r = r - (min as f64).log2();
        [r * theta.cos(), r * theta.sin()]
    };

    bins.map(move |bin| {
        (bin.val, [
            polar(bin.xrange.0, bin.yrange.0, ymin),
            polar(bin.xrange.0, bin.yrange.1, ymin),
            polar(bin.xrange.1, bin.yrange.1, ymin),
            polar(bin.xrange.1, bin.yrange.0, ymin)
        ])
    })
}
fn make_rectangles<'a>(bins: impl Iterator<Item=SlimBin> + 'a) -> impl Iterator<Item=(f32, [[f64;2];4])> + 'a {
    bins.map(|bin| {
            (bin.val, [
                [bin.xrange.0, bin.yrange.0],
                [bin.xrange.0, bin.yrange.1],
                [bin.xrange.1, bin.yrange.1],
                [bin.xrange.1, bin.yrange.0]
            ])
        })
}
