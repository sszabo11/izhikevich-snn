use std::f32::consts::E;

use ndarray::{Array1, Array2};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Uniform, num_traits::Pow},
};

use crate::izhikevich::ModelResponse;

pub struct Logger {
    pub spikes_data: Option<Array2<u32>>,
    pub weights: Option<Array2<f32>>,
    pub v_history: Array2<f32>,       // (timesteps, neurons)
    pub spike_history: Array2<u32>,   // or Vec<Vec<f32>>
    pub weights_history: Array2<f32>, // optional
    pub layer_id: usize,
}

impl Logger {
    pub fn init(T: usize, out_n: usize) -> Self {
        Self {
            spikes_data: None,
            weights: None,
            v_history: Array2::zeros((T * 10, out_n)),
            spike_history: Array2::zeros((T * 10, out_n)),
            weights_history: Array2::zeros((T * 10, out_n)),
            layer_id: 0,
        }
    }
    pub fn set_T(&mut self, T: usize, dt: f32, layer: IzhikevichLayer) {
        self.spikes_data = Some(Array2::zeros(((T as f32 / dt) as usize, layer.out_n)));
        self.weights = Some(Array2::zeros((
            (T as f32 / dt) as usize,
            layer.out_n * layer.n_conns,
        )));
    }
}
pub struct IzhikevichModel {
    pub layers: Vec<IzhikevichLayer>,
}

impl IzhikevichModel {
    pub fn init() -> Self {
        Self { layers: Vec::new() }
    }
    pub fn add_layer(&mut self, mut layer: IzhikevichLayer) {
        layer.id = self.layers.len();
        self.layers.push(layer);
    }

    pub fn run(&mut self, T: usize, input: Array1<f32>) -> Array1<f32> {
        let dt: f32 = 0.1;

        println!("inp1: {}", input);
        let steps = (T as f32 / dt) as usize;
        let n_layers = self.layers.len();
        let mut input_spikes: Array1<f32> = input;
        let mut recent_spikes: Vec<Array1<f32>> = Vec::new();

        for t in 0..steps {
            for idx in 0..n_layers {
                println!("inp: {}", input_spikes);
                let layer = &mut self.layers[idx];
                let output = if idx == 0 {
                    layer.process(t, &input_spikes, dt, &Array1::zeros(layer.out_n))
                } else {
                    layer.process(t, &input_spikes, dt, &recent_spikes[idx - 1])
                };

                if t == 0 {
                    recent_spikes.push(Array1::zeros(layer.out_n));
                }
                recent_spikes[idx] += &output;
                input_spikes = output;
            }
        }
        input_spikes
    }
}

#[derive(Clone)]
pub struct IzhikevichLayerBuilder {
    pub in_n: usize,
    pub out_n: usize,
    pub n_conns: usize,
    pub threshold: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub max_weight: f32,
    pub min_weight: f32,
    pub T: usize,
    pub spike_buffer_size: usize,
}
impl Default for IzhikevichLayerBuilder {
    fn default() -> Self {
        Self {
            in_n: 10,
            out_n: 50,
            n_conns: 20,
            threshold: 30.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            T: 100,
            d: 8.0,
            a_plus: 0.008,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            max_weight: 1.0,
            min_weight: -1.0,
            spike_buffer_size: 256,
        }
    }
}

impl IzhikevichLayerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    // Layer size
    pub fn in_n(mut self, in_n: usize) -> Self {
        self.in_n = in_n;
        self
    }

    pub fn T(mut self, T: usize) -> Self {
        self.T = T;
        self
    }

    pub fn out_n(mut self, out_n: usize) -> Self {
        self.out_n = out_n;
        self
    }

    pub fn n_conns(mut self, n_conns: usize) -> Self {
        self.n_conns = n_conns;
        self
    }

    // Neuron parameters
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn a(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    pub fn b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    pub fn c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    pub fn d(mut self, d: f32) -> Self {
        self.d = d;
        self
    }

    // STDP parameters
    pub fn a_plus(mut self, a_plus: f32) -> Self {
        self.a_plus = a_plus;
        self
    }

    pub fn a_minus(mut self, a_minus: f32) -> Self {
        self.a_minus = a_minus;
        self
    }

    pub fn tau_plus(mut self, tau_plus: f32) -> Self {
        self.tau_plus = tau_plus;
        self
    }

    pub fn tau_minus(mut self, tau_minus: f32) -> Self {
        self.tau_minus = tau_minus;
        self
    }

    pub fn max_weight(mut self, max_weight: f32) -> Self {
        self.max_weight = max_weight;
        self
    }

    pub fn min_weight(mut self, min_weight: f32) -> Self {
        self.min_weight = min_weight;
        self
    }

    pub fn spike_buffer_size(mut self, size: usize) -> Self {
        self.spike_buffer_size = size;
        self
    }

    // Build the layer
    pub fn build(self) -> IzhikevichLayer {
        IzhikevichLayer::from_builder(self)
    }
}

pub struct IzhikevichLayer {
    membranes: Array1<f32>,
    u: Array1<f32>,         // Recovery var for each neuron
    pub conns: Array2<u32>, // Each neuron has conns
    pub weights: Array2<f32>,

    pub spike_history: Vec<Vec<f32>>,

    pub logger: Logger,
    pub recent_spikes: Array1<f32>,
    pub in_n: usize,
    pub out_n: usize,
    pub n_conns: usize,
    pub threshold: f32,
    a: f32, // Time scale of the recovery variable. Smaller -> slower recovery
    b: f32, // Sensitivty of the recovery variable. Larger -> stronger coupling
    c: f32, // Reset voltage of membrane
    d: f32, // Increment of the recovery variable after a spike.

    pub id: usize,
    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
    max_weight: f32,
    min_weight: f32,
}

impl IzhikevichLayer {
    pub fn from_builder(builder: IzhikevichLayerBuilder) -> Self {
        let out_n = builder.out_n;

        Self {
            in_n: builder.in_n,
            out_n,
            n_conns: builder.n_conns,
            threshold: builder.threshold,
            a: builder.a,
            b: builder.b,
            id: 0,
            c: builder.c,
            d: builder.d,
            recent_spikes: Array1::zeros(builder.out_n),
            a_plus: builder.a_plus,
            a_minus: builder.a_minus,
            tau_plus: builder.tau_plus,
            tau_minus: builder.tau_minus,
            max_weight: builder.max_weight,
            min_weight: builder.min_weight,
            logger: Logger::init(builder.T, builder.out_n),
            membranes: Array1::zeros(out_n),
            u: Array1::zeros(out_n),
            conns: Array2::random(
                (out_n, builder.n_conns),
                Uniform::new(0, builder.in_n as u32).unwrap(),
            ),
            weights: Array2::random(
                (out_n, builder.n_conns),
                Uniform::new(builder.min_weight, builder.max_weight).unwrap(),
            ),
            spike_history: vec![vec![]; out_n],
        }
    }

    fn apply_stdp(&mut self, post_i: usize, t_post: f32, recent_spikes: &Array1<f32>) {
        let conns = self.conns.row(post_i);

        for (arr_i, &conn) in conns.iter().enumerate() {
            //let recent_spikes = &self.spike_history[conn as usize];

            for t_pre in recent_spikes.iter() {
                let d_t = t_post - t_pre;

                let mut dw = 0.0;

                if d_t > 0.0 && d_t < self.tau_plus {
                    // LTP
                    dw = self.a_plus * (E * (-d_t / self.tau_plus).exp());
                } else if d_t < 0.0 && d_t > -self.tau_minus {
                    // LTD
                    dw = -self.a_minus * (E * (d_t / self.tau_minus).exp());
                }

                if dw != 0.0 {
                    self.weights[[post_i, arr_i]] += dw;
                    //    self.weights[[post_i, arr_i]] = self.weights[[post_i, arr_i]].clamp(0.0, 1.0);
                }
            }
        }
    }

    pub fn process(
        &mut self,
        t: usize,
        input: &Array1<f32>,
        dt: f32,
        recent_spikes: &Array1<f32>,
    ) -> Array1<f32> {
        //let dt: f32 = 0.1;

        //let mut u_values = Array2::zeros((steps, self.out_n));
        //let mut v_values = Array2::zeros((steps, self.out_n));
        //let mut I_values = Array2::zeros((steps, self.out_n));

        //let mut t_values = Array2::zeros((steps, self.out_n));

        //let t_start = 50.0;
        //let t_end = T;
        let I_baseline: f32 = 0.0;
        let log_interval = 1;

        let mut rng = rand::rng();
        let mut output_spikes = Array1::zeros(self.out_n);
        //for t in 0..steps {
        for i in 0..self.out_n {
            let mut I = I_baseline;
            //let mut I = input[i] + I_baseline;
            //println!("{}", logger.spikes_data);
            //if t != 0 {
            for (j, &conn_n_idx) in self.conns.row(i).iter().enumerate() {
                let weight = self.weights[[i, j as usize]];
                let pre_spike = input[conn_n_idx as usize] as f32;
                let v_incoming = pre_spike * weight;
                if i == 0 && self.id == 1 && t == 0 {
                    //println!("v inc: {} | {} | {}", v_incoming, weight, pre_spike);
                }

                I += v_incoming;
            }
            //}

            //let mut I = rng.random::<f32>() * 20.0;
            let t_ms = t as f32 * dt;

            //if t_ms >= t_start && t_ms <= t_end {
            //    I += I_baseline;
            //}

            self.logger.v_history[[t, i]] = self.membranes[i];
            if self.membranes[i] > self.threshold {
                self.membranes[i] = self.c; // Reset
                self.u[i] += self.d;

                if let Some(data) = &mut self.logger.spikes_data {
                    data[[t, i]] = 1;
                }

                self.logger.spike_history[[t, i]] = 1;

                self.spike_history[i].push(t_ms);

                while !self.spike_history[i].is_empty() && t_ms - self.spike_history[i][0] > 50.0 {
                    self.spike_history[i].remove(0);
                }
                output_spikes[i] = 1.0;

                self.apply_stdp(i, t_ms, recent_spikes);
            }

            let v = self.membranes[i];
            let u = self.u[i];
            self.membranes[i] += dt * 0.5 * (0.04_f32 * v.pow(2) + 5.0 * v + 140_f32 - u + I);
            self.u[i] += dt * self.a * (self.b * v - u);

            //u_values[[t, i]] = u;
            //v_values[[t, i]] = v;
            //I_values[[t, i]] = I;
            //t_values[[t, i]] = t_ms;
            if t != 0 {
                if t % log_interval == 0 {
                    for c in 0..self.n_conns {
                        let conn_idx = c;
                        let global_col = i * self.n_conns + conn_idx;
                        let row = t / log_interval;

                        //self.logger.weights_history[[row, global_col]] =
                        //    self.weights[[i, conn_idx]];

                        if let Some(data) = &mut self.logger.weights {
                            data[[row, global_col]] = self.weights[[i, conn_idx]];
                        }

                        //        logger.weights[[row, global_col]] = self.weights[[i, conn_idx]];
                    }
                }
            }
        }
        //}

        output_spikes
    }
}
