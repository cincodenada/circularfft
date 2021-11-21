use rustfft::{FftPlanner, num_complex::Complex};
use itertools::Itertools;

pub type Freq = f32;
pub type Mag = f32;
pub type Point = Complex<Mag>;

pub enum Window {
  Square,
  Hann,
  Hamming,
  Blackman
}
impl Window {
    fn apply(&self, samples: Vec<f32>) -> Vec<f32> {
        let N = samples.len() as f32;
        samples.into_iter().enumerate().map(
            |(n, v)| self.getVal(N, n as f32, v)
        ).collect()
    }

    fn getVal(&self, N: f32, n: f32, v: f32) -> f32 {
        let CosineSum = |a0: f32, a1: f32, a2: f32| {
            move |N, n, v| {
                let term: f32 = 2.0*std::f32::consts::PI*n/N;
                a0 - a1*term.cos() + a2*(2.0*term).cos()
            }
        };
        let SimpleCosine = move |a0: f32| CosineSum(a0, 1.0-a0, 0.0);
        let SquareWin = |N, n, v| v;

        match(self) {
            Square => SquareWin(N, n, v),
            Hann => SimpleCosine(0.5)(N, n, v),
            Hamming => SimpleCosine(25.0/46.0)(N, n, v),
            Blackman => CosineSum(0.42, 0.5, 0.08)(N, n, v)
        }
    }
}

pub struct Bin {
    pub val: Point,
    pub freq: Freq,
    pub mag: Mag,
    pub freq_whole: Freq,
    pub freq_fract: Freq
}
impl Bin {
    fn from_sample(idx: usize, val: Point, half_size: usize, max_freq: Freq) -> Bin {
        let freq = (idx as Freq/half_size as Freq)*max_freq;
        let freq_log = freq.log2();
        Bin {
            freq, val,
            freq_whole: freq_log.floor(),
            freq_fract: freq_log.fract(),
            mag: val.norm()
        }
    }
}
impl std::fmt::Debug for Bin {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_fmt(format_args!("Bin @ {}, Mag {} ({}/{})", self.freq, self.mag, self.val.re, self.val.im))
    }
}

pub struct Spectrogram {
    pub sample_rate: u32,
    pub fft_size: usize,
    pub half_size: usize,
    pub columns: Vec<Column>,
    pub max_freq: Freq,
    pub max_mag: Mag,
    pub min_mag: Mag,
}
impl Spectrogram {
    pub fn from_samples(samples: &Vec<f32>, fft_size: usize, overlap: f32, windowFunc: Window, sample_rate: u32, channels: u16) -> Spectrogram {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut min_mag = f32::INFINITY;
        let mut max_mag = f32::NEG_INFINITY;
        let step = (fft_size as f32 * overlap) as usize;

        let samples = samples.chunks(channels as usize)
            .map(|v| v.into_iter().sum::<f32>()/v.len() as f32)
            .map(Point::from).collect::<Vec<_>>();

        let columns = (0..samples.len()-fft_size).step_by(step).map(|start| {
            let mut buffer = samples[start..start + fft_size].to_vec();
            fft.process(&mut buffer);
            let col = Column::from_bins(sample_rate, buffer);
            if col.min_mag < min_mag { min_mag = col.min_mag }
            if col.max_mag > max_mag { max_mag = col.max_mag }
            col
        }).collect();

        Spectrogram {
            sample_rate,
            fft_size,
            columns,
            min_mag,
            max_mag,
            half_size: fft_size/2,
            max_freq: (sample_rate/2) as Freq
        }
    }
}

#[derive(Debug)]
pub struct Column {
    pub bins: Vec<Bin>,
    pub max_mag: f32,
    pub min_mag: f32
}
impl Column {
    pub fn from_bins(sample_rate: u32, bins: Vec<Point>) -> Column {
        let mut min_mag = f32::INFINITY;
        let mut max_mag = f32::NEG_INFINITY;
        let fft_size = bins.len();
        let half_size = fft_size/2;
        let max_freq = sample_rate as Freq/2.0;
        let bins = bins.into_iter().take(half_size).enumerate()
            .map(|(idx, v)| {
                let bin = Bin::from_sample(idx, v, half_size, max_freq);
                if bin.mag < min_mag { min_mag = bin.mag }
                if bin.mag > max_mag { max_mag = bin.mag }
                bin
            }).collect();

         Column { bins, min_mag, max_mag }
    }
}


