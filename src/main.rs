extern crate piston_window;
extern crate tuple;

mod bracketed_chunks;
mod spectrogram;

use bracketed_chunks::*;
use spectrogram as spec;

use piston_window::*;
use tuple::*;

use druid::widget::{Button, Flex, Label};
use druid::{AppLauncher, LocalizedString, PlatformError, Widget, WidgetExt, WindowDesc, Data};

use std::env;
use std::time::{Duration, SystemTime};
use std::fs::File;
use std::path::Path;
use wav::BitDepth;
use std::convert::TryFrom;
use inline_python::python;
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

    fn map(&self, val: spec::Freq) -> [f32;4] {
        match (val.log2() - self.min)/self.halfrange {
            v @ 0.0..=1.0 => [0.0, v/2.0, 0.0, 1.0],
            v => [v-1.0, v/2.0, 0.0, 1.0],
        }
    }
}

fn main() -> Result<(), std::io::Error> {
    let args = env::args().collect::<Vec<_>>();
    let fftpow = args[1].parse::<u32>().unwrap();
    let overlap = args[2].parse::<f32>().unwrap();
    let speed = args[3].parse::<f32>().unwrap();

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
        _ => panic!("Ack!"),
        BitDepth::Empty => panic!("Ack!")
    };
    dbg!(&header);

    //let floatMax = |a:f32, b:f32| max(OrderedFloat(a), OrderedFloat(b)).into();

    let spectrogram = spec::Spectrogram::from_samples(
        &samples, header.sampling_rate, header.channel_count
    ).calculate_with(fftsize, overlap, spec::Window::Hann);
    let ms_per_col = (((fftsize as f32 * (1.0-overlap))/header.sampling_rate as f32)/speed*1000.0) as u64;
    dbg!(ms_per_col);

    let colorer = Colorer::new(spectrogram.min_mag, spectrogram.max_mag);

    make_druid_window(&spectrogram, colorer, Duration::from_millis(ms_per_col.into()));
    //make_window(&spectrogram, colorer, Duration::from_millis(ms_per_col.into()));
    //make_circle_plot(&spectrogram);
    //make_rect_plot(&spectrogram);

    Ok(())
}

#[derive(Clone, Data)]
struct Counter(i32);

fn make_druid_window(spectrogram: &spec::Spectrogram, colorer: Colorer<spec::Freq>, spf: std::time::Duration) {
    // Window builder. We set title and size
    let main_window = WindowDesc::new(build_druid_window)
        .title("Spectrogram Toy")
        .window_size((200.0, 100.0));

    // Data to be used in the app (=state)
    let data = Counter(0);

    // Run the app
    AppLauncher::with_window(main_window)
        .use_simple_logger() // Neat!
        .launch(data);
}

fn build_druid_window() -> impl Widget<Counter> {
    // The label text will be computed dynamically based on the current locale and count
    let text = LocalizedString::new("hello-counter")
        .with_arg("count", |data: &Counter, _env| (*data).0.into());
    let label = Label::new(text).padding(5.0).center();

    // Two buttons with on_click callback
    let button_plus = Button::new("+1")
        .on_click(|_ctx, data: &mut Counter, _env| (*data).0 += 1)
        .padding(5.0);
    let button_minus = Button::new("-1")
        .on_click(|_ctx, data: &mut Counter, _env| (*data).0 -= 1)
        .padding(5.0);

    // Container for the two buttons
    let flex = Flex::row()
        .with_child(button_plus)
        .with_spacer(1.0)
        .with_child(button_minus);

    // Container for the whole UI
    Flex::column()
        .with_child(label)
        .with_child(flex)
}

fn make_window(spectrogram: &spec::Spectrogram, colorer: Colorer<spec::Freq>, spf: std::time::Duration) {
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
                if(v > spf) {
                    col = slice.next().unwrap();
                    time = SystemTime::now();
                }
            }

            clear([0.5, 0.5, 0.5, 1.0], g);
            let rects = make_wedges(col, freq_range);
            let dims = c.viewport.unwrap().draw_size.map(f64::from);
            rects.into_iter().map(|(val, points)| polygon(
                colorer.map(val),
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
    let (x, y, values) = make_color_mesh(&mag, &freqbins, &onefreq, xbinsf);

    python! {
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        x = [row + [math.pi*2] for row in 'x]
        x = x + [x[-1]]
        y = [col + [col[-1]] for col in 'y] + [['y[-1][0]+1] * (len('y[0])+1)]
        def dims(x):
            print(len(x))
            print([len(r) for r in x])
        dims(x)
        dims(y)
        dims('values)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.set_rmax(3)
        ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.set_xticks([(s+0.5)/12*math.pi*2 for s in range(0,12)])
        ax.set_xticklabels(['|']*12)
        ax.grid(True)

        plt.pcolormesh(x, y, 'values)
        plt.show()
    }
}

fn make_rect_plot(spectrogram: &spec::Spectrogram) {
    //let starts: Vec<usize> = (0..width).map(|v| v*fftsize/2).collect();
    //let time: Vec<usize> = starts.iter().map(|start| vec![*start+fftsize/4;fftsize/2]).flatten().collect();
    //let freqbins: Vec<f32> = (0..fftsize/2).map(|v| floatMax((v as f32).log10(),0.0)).collect::<Vec<_>>();
    //let freq: Vec<f32> = starts.iter().map(|_| freqbins.to_vec()).flatten().collect();

    let cols = spectrogram.columns.iter().enumerate();
    let mut vals = cols.clone().skip(1).map(|(idx, c)| c.bins.iter().skip(1)
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

    python! {
        import matplotlib.pyplot as plt

        plt.hist2d('time, 'freq, ['bincount, 'freqbins],weights='mag)
        plt.show()
    }

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
            Some([min, max]) if comp >= *max => {
                curfreq = freqiter.next();
                curcol = col.next().unwrap();
            },
            _ => {}
        }
        (*frac * std::f32::consts::PI*2.0, *curcol)
    }).unzip()).unzip();

    (dupcol.0, wholes, dupcol.1)
}

fn rep_last<T>(v: &Vec<T>) -> impl Iterator<Item=&T> {
    v.iter().chain(std::iter::once(&v[v.len()-1]))
}
fn add_pi(v: &Vec<f32>) -> impl Iterator<Item=&f32> {
    v.iter().chain(std::iter::once(&(std::f32::consts::PI*2.0)))
}

fn bracket<X: Copy, V: Copy>(it: impl Iterator<Item=(X, V)>, min: X, max: X) -> impl Iterator<Item=(X, V)> {
    it.enumerate()
        .map(move |(idx, tup)| match idx {
            0 => vec![(min, tup.1), tup],
            _ => vec![tup]
        }).flatten()
}

fn dbgIter<I, T>(it: I) -> impl Iterator<Item=T> where I: Iterator<Item=T>, T: std::fmt::Debug {
    let collected = it.collect::<Vec<_>>();
    dbg!(&collected);
    collected.into_iter()
}

fn make_bins<'a>(col: &'a spec::Column, clip: FreqRange) -> impl Iterator<Item=(spec::Mag, FreqRange, OctRange)> + 'a {
    let sharder = OctaveSharder { min: clip.0, max: clip.1 };
    let bounds = (0.0, 1.0);
    col.bins.iter().skip(1)
        .filter(move |bin| bin.freq >= clip.0 && bin.freq < clip.1)
        .bracketed_chunks(sharder)
        .tuple_windows()
        .map(move |v| {
            match v {
                (ShardResult::Start(cur_shard, cur), ShardResult::Start(next_shard, next)) => vec![
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
                (ShardResult::Start(prev_shard, prev), ShardResult::Item(cur_shard, cur)) => vec![
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
                (ShardResult::Item(prev_shard, prev), ShardResult::Item(cur_shard, cur)) => vec![
                    (
                        cur.mag,
                        (prev.freq_fract, cur.freq_fract),
                        (cur_shard, cur_shard + 1)
                    )
                ],
                (ShardResult::Item(prev_shard, prev), ShardResult::Start(cur_shard, cur)) => vec![
                    (
                        cur.mag,
                        (prev.freq_fract, bounds.1),
                        (cur_shard, cur_shard + 1)
                    )
                ],
                _ => vec![]
            }
        }).flatten()
}

fn make_wedges<'a>(col: &'a spec::Column, clip: FreqRange) -> impl Iterator<Item=(f32, [[f64;2];4])> + 'a {
    let polar = move |thetaish, r| {
        let theta: f64 = thetaish as f64*2.0*std::f64::consts::PI;
        let r = (r as f32 - clip.0.log2()) as f64;
        [r * theta.cos(), r * theta.sin()]
    };
    make_bins(col, clip)
        .map(move |(m, x, y)| (m, [
            polar(x.0, y.0),
            polar(x.0, y.1),
            polar(x.1, y.1),
            polar(x.1, y.0)
        ]))
}
fn make_rectangles<'a>(col: &'a spec::Column, clip: FreqRange) -> impl Iterator<Item=(f32, [[f64;2];4])> + 'a {
    make_bins(col, clip)
        .map(|(m, x, y)| {
            let x = x.map(|v| v as f64);
            let y = y.map(|v| v as f64);
            (m, [
                [x.0, y.0],
                [x.0, y.1],
                [x.1, y.1],
                [x.1, y.0]
            ])
        })
}
