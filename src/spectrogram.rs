use rustfft::{FftPlanner, num_complex::Complex};
use itertools::Itertools;

pub type Freq = f32;
pub type Mag = f32;
pub type Point = Complex<Mag>;

pub struct Bin {
    pub val: Point,
    pub freq: Freq,
    pub mag: Mag,
    pub freq_log: Freq
}
impl Bin {
    fn from_sample(idx: usize, val: Point, half_size: usize, max_freq: Freq) -> Bin {
        let freq = (idx as Freq/half_size as Freq)*max_freq;
        Bin {
            freq, val,
            freq_log: freq.log2(),
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
    pub fn from_samples(samples: &Vec<f32>, fft_size: usize, sample_rate: u32, channels: u16) -> Spectrogram {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut min_mag = f32::INFINITY;
        let mut max_mag = f32::NEG_INFINITY;
        let columns = samples.chunks(channels as usize)
            .map(|v| v.into_iter().sum::<f32>()/v.len() as f32)
            .map(Point::from)
            .chunks(fft_size).into_iter()
            .filter_map(|samples| {
                let mut buffer = samples.collect::<Vec<_>>();
                if(buffer.len() < fft_size) { return None; }
                fft.process(&mut buffer);
                let col = Column::from_bins(sample_rate, buffer);
                if col.min_mag < min_mag { min_mag = col.min_mag }
                if col.max_mag > max_mag { max_mag = col.max_mag }
                Some(col)
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


