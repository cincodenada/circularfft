use rustfft::{FftPlanner, num_complex::Complex};
use std::fs::File;
use std::path::Path;
use wav::BitDepth;
use std::convert::TryFrom;
use inline_python::python;

fn main() -> Result<(), std::io::Error> {
    let fftsize = 8192;

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

    let width=100;

    let mut buffer = complex[10000..18192].to_vec();
    fft.process(&mut buffer);
    let mag = buffer.into_iter().map(Complex::norm).collect::<Vec<_>>();
    let freq = (0..8192).map(|v| f64::try_from(v).unwrap()).collect::<Vec<f64>>();
    let starts: Vec<usize> = (0..width).map(|v| v*fftsize).collect();
    let mag: Vec<f32> = starts.iter().map(|start| {
        complex[*start..start+fftsize].to_vec().into_iter().map(Complex::norm).collect::<Vec<f32>>()
    }).flatten().collect();
    let time: Vec<usize> = starts.iter().map(|start| vec![*start;fftsize]).flatten().collect();
    let freq: Vec<usize> = starts.iter().map(|_| (0..fftsize).collect::<Vec<usize>>()).flatten().collect();

    python! {
        import matplotlib.pyplot as plt
        plt.hist2d('time, 'freq, ['width, 'fftsize],weights='mag)
        plt.show()
    }

    Ok(())
}
