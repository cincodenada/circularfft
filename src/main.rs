extern crate piston_window;
extern crate tuple;
use piston_window::*;
use tuple::*;

use rustfft::{FftPlanner, num_complex::Complex};
use std::fs::File;
use std::path::Path;
use wav::BitDepth;
use std::convert::TryFrom;
use inline_python::python;
use std::cmp::max;
use ordered_float::OrderedFloat;
use itertools::Itertools;

fn main() -> Result<(), std::io::Error> {
    let fftsize = 2_usize.pow(4);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fftsize);

    let mut inp_file = File::open(Path::new("input.wav"))?;
    let (header, data) = wav::read(&mut inp_file)?;

    type FftPoint = Complex<f32>;

    let complex : Vec<FftPoint> = match data {
        //BitDepth::Sixteen(vec) => vec.into_iter().collect(),
        //BitDepth::TwentyFour(vec) => vec.into_iter().collect(),
        //BitDepth::ThirtyTwoFloat(vec) => vec.into_iter().collect(),
        BitDepth::ThirtyTwoFloat(vec) => vec.into_iter().map(FftPoint::from).collect(),
        _ => panic!("Ack!"),
        BitDepth::Empty => panic!("Ack!")
    };

    //let floatMax = |a:f32, b:f32| max(OrderedFloat(a), OrderedFloat(b)).into();

    let width=complex.len()/fftsize*2-1;

    let starts: Vec<usize> = (0..width).map(|v| v*fftsize/2).collect();
    let mag: Vec<Vec<f32>> = starts.iter().map(|start| {
        let mut buffer = complex[*start..start+fftsize].to_vec();
        fft.process(&mut buffer);
        buffer.into_iter().take(fftsize/2).map(|v| v.norm().log2()).collect::<Vec<f32>>()
    }).collect();
    let time: Vec<Vec<usize>> = starts.iter().map(|start| vec![*start+fftsize/4;fftsize/2]).collect();

    let freq = (0..fftsize).map(|v| (v as f64)).collect::<Vec<f64>>();
    let freqbins: Vec<f32> = (1..fftsize/2).map(|v| (v as f32).log2()).collect::<Vec<_>>();

    let r: Vec<f32> = freqbins.iter().map(|v| v.floor()).collect();
    let theta = r.iter().zip(freqbins.iter()).map(|(&r, &v)| v - r).collect::<Vec<_>>();

    let xbins = r.to_vec().into_iter().map(OrderedFloat).max().unwrap();
    let xbinsf: f32 = xbins.into();
    let onefreq = freqbins.iter().filter(|f| OrderedFloat(**f) >= xbins).map(|f| f - xbinsf).collect::<Vec<_>>();

    //let flatmag = mag.into_iter().flatten().collect::<Vec<_>>();
    //let flattime = time.into_iter().flatten().collect::<Vec<_>>();

    let (x, y, values) = make_color_mesh(&mag[200], &freqbins, &onefreq, xbinsf);
    dbg!(make_rectangles(&freq,&mag[200],(16.35, 7902.13)));
    //let mut window: PistonWindow =
    //    WindowSettings::new("Hello World!", [512; 2])
    //        .build().unwrap();
    //while let Some(e) = window.next() {
    //    window.draw_2d(&e, |c, g, _| {
    //        clear([0.5, 0.5, 0.5, 1.0], g);
    //        make_rectangles(&x,&y,&values).map(|(color, points)| polygon(color, points, c.transform, g))
    //    });
    //}

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

fn make_rectangles(freq: &[f64], mag: &[f32], clip: (f64, f64)) -> Vec<(Vec<f32>, Vec<Vec<f64>>)> {
    let clip_ord = clip.map(OrderedFloat);
    let clipped_freqs = freq.iter().map(|f| OrderedFloat(*f)).zip(mag.iter())
        .filter(|(f, _)| *f > clip_ord.0 && *f < clip_ord.1)
        .map(|(f, m)| (f.into(), m));
    dbg!(&freq);
    let boxed = std::iter::once((clip.0, &mag[0]))
        .chain(clipped_freqs)
        .chain(std::iter::once((clip.1, &mag[mag.len()-1])));

    boxed.tuple_windows().map(|((f, m), (nextf, nextm))| (
        vec![1.0,0.0,0.0,1.0],
        vec![
            vec![f.into(), f.floor()],
            vec![f.into(), f.floor()+1.0],
            vec![nextf.into(), f.floor()+1.0],
            vec![nextf.into(), f.floor()]
        ].into_iter().map(|v| v.into_iter().map(|f| f as f64).collect()).collect()
    )).collect()
}
