use rustfft::{FftPlanner, num_complex::Complex};
use std::fs::File;
use std::path::Path;
use wav::BitDepth;

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

    let mut buffer = complex[10000..18192].to_vec();
    fft.process(&mut buffer);
    dbg!("{}", buffer.into_iter().map(Complex::norm).collect::<Vec<_>>());
    Ok(())
}
