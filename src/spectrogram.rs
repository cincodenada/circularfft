use rustfft::{num_complex::Complex};

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

#[derive(Debug)]
pub struct Column {
    pub sample_rate: u32,
    pub fft_size: usize,
    pub half_size: usize,
    pub bins: Vec<Bin>,
    pub max_freq: Freq,
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

         Column { sample_rate, fft_size, half_size, max_freq, bins, min_mag, max_mag }
    }
}


