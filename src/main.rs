use rustfft::{FftPlanner, num_complex::Complex};
use std::fs::File;
use std::path::Path;
use wav::BitDepth;
use std::convert::TryFrom;
use inline_python::python;
use std::cmp::max;
use ordered_float::OrderedFloat;

fn main() -> Result<(), std::io::Error> {
    let fftsize = 2_usize.pow(14);

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

    let floatMax = |a:f32, b:f32| max(OrderedFloat(a), OrderedFloat(b)).into();

    let width=complex.len()/fftsize*2;

    let freq = (0..fftsize).map(|v| (v as f64)).collect::<Vec<f64>>();
    let starts: Vec<usize> = (0..width).map(|v| v*fftsize/2).collect();
    let mag: Vec<Vec<f32>> = starts.iter().map(|start| {
        let mut buffer = complex[*start..start+fftsize].to_vec();
        fft.process(&mut buffer);
        buffer.into_iter().take(fftsize/2).map(|v| v.norm().log10()).collect::<Vec<f32>>()
    }).collect();
    let time: Vec<Vec<usize>> = starts.iter().map(|start| vec![*start+fftsize/4;fftsize/2]).collect();
    let freqbins: Vec<f32> = (0..fftsize/2).map(|v| floatMax((v as f32).log10(),0.0)).collect::<Vec<_>>();

    let freq: Vec<f32> = starts.iter().map(|_| freqbins.to_vec()).flatten().collect();

    let flatmag = mag.into_iter().flatten().collect::<Vec<_>>();
    let flattime = time.into_iter().flatten().collect::<Vec<_>>();

    python! {
        import matplotlib.pyplot as plt
        plt.hist2d('flattime, 'freq, ['starts, 'freqbins],weights='flatmag)
        plt.show()
    }

    Ok(())
}
