extern crate piston_window;
extern crate tuple;

mod bracketed_chunks;
mod spectrogram;

use bracketed_chunks::*;
use spectrogram as spec;

use piston_window::*;
use tuple::*;

use rustfft::{FftPlanner, num_complex::Complex};
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
            v => Some(v.freq_log.floor() as usize)
        }
    }
}

fn main() -> Result<(), std::io::Error> {
    let fftsize = 2_usize.pow(14);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fftsize);

    let mut inp_file = File::open(Path::new("input.wav"))?;
    let (header, data) = wav::read(&mut inp_file)?;

    let complex : Vec<spec::Point> = match data {
        //BitDepth::Sixteen(vec) => vec.into_iter().collect(),
        //BitDepth::TwentyFour(vec) => vec.into_iter().collect(),
        //BitDepth::ThirtyTwoFloat(vec) => vec.into_iter().collect(),
        // TODO: We probably shouldn't need to collect() here
        BitDepth::ThirtyTwoFloat(vec) => vec
            .chunks(header.channel_count as usize)
            .map(|v| v.into_iter().sum::<f32>()/v.len() as f32)
            .map(spec::Point::from)
            .collect(),
        _ => panic!("Ack!"),
        BitDepth::Empty => panic!("Ack!")
    };
    dbg!(complex.len());
    let sample_rate = header.sampling_rate;
    let max_freq = sample_rate/2;
    dbg!(&header);

    //let floatMax = |a:f32, b:f32| max(OrderedFloat(a), OrderedFloat(b)).into();

    let width=complex.len()/fftsize*2-1;

    let mut absmin = f32::INFINITY;
    let mut absmax = f32::NEG_INFINITY;
    let starts: Vec<usize> = (0..width).map(|v| v*fftsize/2).collect();
    let mag: Vec<spec::Column> = starts.iter().map(|start| {
        let mut buffer = complex[*start..start+fftsize].to_vec();
        fft.process(&mut buffer);
        let col = spec::Column::from_bins(sample_rate, buffer);
        if col.min_mag < absmin { absmin = col.min_mag }
        if col.max_mag > absmax { absmax = col.max_mag }
        col
    }).collect();
    //dbg!(&mag);
    let time: Vec<Vec<usize>> = starts.iter().map(|start| vec![*start+fftsize/4;fftsize/2]).collect();

    let freqbins: Vec<f32> = (1..fftsize/2).map(|v| (v as f32).log2()).collect::<Vec<_>>();

    let r: Vec<f32> = freqbins.iter().map(|v| v.floor()).collect();
    let theta = r.iter().zip(freqbins.iter()).map(|(&r, &v)| v - r).collect::<Vec<_>>();

    let xbins = r.to_vec().into_iter().map(OrderedFloat).max().unwrap();
    let xbinsf: f32 = xbins.into();
    let onefreq = freqbins.iter().filter(|f| OrderedFloat(**f) >= xbins).map(|f| f - xbinsf).collect::<Vec<_>>();

    //let flatmag = mag.into_iter().flatten().collect::<Vec<_>>();
    //let flattime = time.into_iter().flatten().collect::<Vec<_>>();

    let makeColorer = |min: spec::Freq, max: spec::Freq| {
        let halfrange = (max.log2()-min.log2())/2.0;
        move |val: f32| match (val.log2()-min.log2())/halfrange {
            v @ 0.0..=1.0 => [0.0, v/2.0, 0.0, 1.0],
            v => [v-1.0, v/2.0, 0.0, 1.0],
        }
    };

    //let (x, y, values) = make_color_mesh(&mag[100], &freqbins, &onefreq, xbinsf);

    let freq_range = (16.35, 8372.02);
    let mapped_range = freq_range.map(|v| (v as f64).log2());
    let mapped_span = mapped_range.1 - mapped_range.0;

    let mut slice = mag.iter().cycle();
    let colorer = makeColorer(absmin, absmax);

    let mut window: PistonWindow =
        WindowSettings::new("Hello World!", [512; 2])
            .build().unwrap();
    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g, _| {
            clear([0.5, 0.5, 0.5, 1.0], g);
            let col = slice.next().unwrap();
            let rects = make_rectangles(col, freq_range);
            let dims = c.viewport.unwrap().draw_size.map(f64::from);
            rects.into_iter().map(|(val, points)| polygon(
                colorer(val),
                &points.map(|p| [p[0]*dims[0], (p[1]-mapped_range.0)/mapped_span*dims[1]]),
                c.transform, g
            )).last();
        });
    }

    //dbg!(&theta);
    //dbg!(&r);
    //dbg!(&xbinsf);
    //dbg!(&onefreq);
    //dbg!(&freqbins);
    //dbg!(&mag[200]);
    //dbg!(&dupcol);
    //dbg!(&wholes);

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

    Ok(())
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

fn make_rectangles(col: &spec::Column, clip: (f32, f32)) -> Vec<(f32, [[f64;2];4])> {
    let sharder = OctaveSharder { min: clip.0, max: clip.1 };
    let bounds = (0.0, 1.0);
    col.bins.iter().skip(1)
        .filter(|bin| bin.freq >= clip.0 && bin.freq < clip.1)
        .bracketed_chunks(sharder)
        .tuple_windows()
        .map(|v| {
            match v {
                (ShardResult::Start(cur_shard, cur), ShardResult::Start(next_shard, next)) => vec![
                    (
                        cur.mag,
                        [
                            [bounds.0, cur_shard as f32],
                            [bounds.0, next_shard as f32],
                            [cur.freq.fract(), next_shard as f32],
                            [cur.freq.fract(), cur_shard as f32]
                        ]
                    ),
                    (
                        cur.mag,
                        [
                            [cur.freq.fract(), cur_shard as f32],
                            [cur.freq.fract(), next_shard as f32],
                            [bounds.1, next_shard as f32],
                            [bounds.1, cur_shard as f32]
                        ]
                    )
                ],
                (ShardResult::Start(prev_shard, prev), ShardResult::Item(cur_shard, cur)) |
                (ShardResult::Item(prev_shard, prev), ShardResult::Item(cur_shard, cur)) => vec![
                    (
                        cur.mag,
                        [
                            [prev.freq.fract(), cur_shard as f32],
                            [prev.freq.fract(), (cur_shard + 1) as f32],
                            [cur.freq.fract(), (cur_shard + 1) as f32],
                            [cur.freq.fract(), cur_shard as f32]
                        ]
                    )
                ],
                _ => vec![]
            }.into_iter().map(|(m, rect)| (m, rect.map(|p| p.map(|c| c as f64))))
        }).flatten().collect()
}
