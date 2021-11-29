// vim: ts=2:sw=0:sts=0:
use rustfft::{FftPlanner, num_complex::Complex};
use itertools::Itertools;

use druid::{Data, Lens};

pub type Freq = f32;
pub type Mag = f32;
pub type Point = Complex<Mag>;

trait WindowFunc {
  #[allow(non_snake_case)]
	fn func(&self, N: f32, n: f32) -> f32;
}
struct CosineSumFunc { a0: f32, a1: f32, a2: f32 }
impl CosineSumFunc {
  fn new(a0: f32, a1: f32, a2: f32) -> Self { Self { a0, a1, a2 } }
  fn simple(a0: f32) -> Self { Self::new(a0, 1.0-a0, 0.0) }
}
impl WindowFunc for CosineSumFunc {
  #[allow(non_snake_case)]
	fn func(&self, N: f32, n: f32) -> f32 {
      let term: f32 = 2.0*std::f32::consts::PI*n/N;
      self.a0 - self.a1*term.cos() + self.a2*(2.0*term).cos()
  }
}
struct IdentityFunc {}
impl WindowFunc for IdentityFunc {
	fn func(&self, _: f32, _: f32) -> f32 { 1.0 }
}

#[derive(Clone, Eq, PartialEq, Data, Copy)]
pub enum Window {
	Square,
	Hann,
	Hamming,
	Blackman
}
#[allow(non_snake_case)]
impl Window {
	fn apply(&self, samples: Vec<Point>) -> Vec<Point> {
		let N = samples.len() as f32;
    let window = self.get_func();
		samples.into_iter().enumerate().map(
			|(n, v)| (v.re * window.func(N, n as f32)).into()
		).collect()
	}

  fn get_func(&self) -> Box<dyn WindowFunc> {
		match self {
			Self::Square => Box::new(IdentityFunc{}),
			Self::Hann => Box::new(CosineSumFunc::simple(0.5)),
			Self::Hamming => Box::new(CosineSumFunc::simple(25.0/46.0)),
			Self::Blackman => Box::new(CosineSumFunc::new(0.42, 0.5, 0.08))
		}
  }
}
impl Default for Window {
	fn default() -> Window { Self::Hann }
}

#[derive(Clone)]
pub struct Bin {
	pub val: Point,
	pub freq: Freq,
	pub freq_range: (Freq, Freq),
	pub mag: Mag,
	pub freq_whole: Freq,
	pub freq_fract: Freq
}
impl Bin {
	fn from_sample(idx: usize, val: Point, half_size: usize, max_freq: Freq) -> Self {
		let binwidth = max_freq/half_size as Freq;
		let freq = idx as Freq*binwidth;
		let freq_log = freq.log2();
		Bin {
			freq, val,
			freq_range: (freq - binwidth/2.0, freq + binwidth/2.0),
			freq_whole: freq_log.floor(),
			freq_fract: freq_log.fract(),
			mag: val.norm()
		}
	}

	fn from_mag(mag: Mag, freq_range: (Freq, Freq)) -> Self {
		Bin {
			freq_range,
			mag,
			val: Complex{re: mag, im: 0.0},
			freq: 0.0,
			freq_whole: 0.0,
			freq_fract: 0.0
		}

	}
}
impl std::fmt::Debug for Bin {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		f.write_fmt(format_args!("Bin @ {} ({}-{}), Mag {} ({}/{})", self.freq, self.freq_range.0, self.freq_range.1, self.mag, self.val.re, self.val.im))
	}
}

// TODO: We shouldn't need Data in here
#[derive(Default, Clone, Copy, Data)]
pub struct Params {
	pub fft_size: usize,
	pub overlap: f32,
	pub window_type: Window,
}

#[derive(Default)]
pub struct Spectrogram {
	pub params: Params,
	pub sample_rate: u32,
	pub half_size: usize,
	pub columns: Vec<Column>,
	pub max_freq: Freq,
	pub max_mag: Mag,
	pub min_mag: Mag,
	samples: Vec<Point>
}
impl Spectrogram {
	pub fn from_samples(samples: &Vec<f32>, sample_rate: u32, channels: u16) -> Spectrogram {
		let samples = samples.chunks(channels as usize)
			.map(|v| v.into_iter().sum::<f32>()/v.len() as f32)
			.map(Point::from).collect::<Vec<_>>();

		Spectrogram {
			sample_rate,
			samples,
			max_freq: (sample_rate/2) as Freq,
			..Spectrogram::default()
		}
	}
	
	pub fn calculate_with(&mut self, params: Params) -> () {
		self.params = params;
		self.calculate();
	}

	pub fn calculate(&mut self) -> () {
		let mut planner = FftPlanner::new();
		let fft = planner.plan_fft_forward(self.params.fft_size);
		let step = (self.params.fft_size as f32 * (1.0-self.params.overlap)) as usize;

		self.min_mag = f32::INFINITY;
		self.max_mag = f32::NEG_INFINITY;

		self.columns = (0..self.samples.len()-self.params.fft_size).step_by(step).map(|start| {
			let mut buffer = self.params.window_type.apply(
        self.samples[start..start + self.params.fft_size].to_vec()
      );
			fft.process(&mut buffer);
			let col = Column::from_bins(self.sample_rate, buffer, (start as f64/self.sample_rate as f64, ((start+step) as f64)/self.sample_rate as f64));
			if col.min_mag < self.min_mag { self.min_mag = col.min_mag }
			if col.max_mag > self.max_mag { self.max_mag = col.max_mag }
			col
		}).collect();
	}
}

#[derive(Debug, Clone)]
pub struct Column {
	pub bins: Vec<Bin>,
	pub max_mag: f32,
	pub min_mag: f32,
	pub time_range: (f64, f64)
}
impl Column {
	pub fn from_bins(sample_rate: u32, bins: Vec<Point>, time_range: (f64, f64)) -> Column {
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

		 Column { bins, min_mag, max_mag, time_range }
	}
}


