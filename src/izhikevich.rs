#![allow(non_snake_case)]
use std::f32::consts::E;

use ndarray::{Array1, Array2};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Uniform, num_traits::Pow},
};
use rand::RngExt;

pub struct ModelResponse {
    pub u_values: Array2<f32>,
    pub v_values: Array2<f32>,
    pub I_values: Array2<f32>,
    pub t_values: Array2<f32>,
    pub threshold: f32,
}

pub struct Logger {
    pub spikes_data: Array2<u32>,
    pub weights: Array2<f32>,
}

impl Logger {
    pub fn new(T: usize, dt: f32, net: &Izhikevich) -> Self {
        Self {
            spikes_data: Array2::zeros(((T as f32 / dt) as usize, net.n_neurons)),
            weights: Array2::zeros((T * 10, net.n_neurons * net.n_conns)),
        }
    }
}

pub struct Izhikevich {
    membranes: Array1<f32>,
    u: Array1<f32>,         // Recovery var for each neuron
    pub conns: Array2<u32>, // Each neuron has conns
    pub weights: Array2<f32>,

    //pub spikes: Array2<u32>,
    pub spike_history: Vec<Vec<f32>>,

    pub n_neurons: usize,
    pub n_conns: usize,
    pub threshold: f32,
    pub a: f32, // Time scale of the recovery variable. Smaller -> slower recovery
    pub b: f32, // Sensitivty of the recovery variable. Larger -> stronger coupling
    pub c: f32, // Reset voltage of membrane
    pub d: f32, // Increment of the recovery variable after a spike.

    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub max_weight: f32,
    pub min_weight: f32,
}

impl Izhikevich {
    pub fn new(
        n_neurons: usize,
        n_conns: usize,
        threshold: f32,
        tau_plus: f32,
        tau_minus: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        T: usize,
    ) -> Self {
        Self {
            n_neurons,
            a,
            b,
            c,
            d,
            tau_plus,
            tau_minus,
            a_plus: 0.01,
            a_minus: 0.02,
            spike_history: vec![],
            threshold,
            min_weight: -1.0,
            max_weight: 1.0,
            n_conns,
            membranes: Array1::zeros(n_neurons),
            //spikes: Array2::zeros((T * 10, n_neurons)),
            u: Array1::zeros(n_neurons),
            weights: Array2::random(
                (n_neurons, n_conns as usize),
                Uniform::new(-1.0, 1.0).unwrap(),
            ),
            conns: Array2::random(
                (n_neurons, n_conns as usize),
                Uniform::new(0, n_neurons as u32).unwrap(),
            ),
        }
    }

    fn apply_stdp(&mut self, post_i: usize, t_post: f32) {
        let conns = self.conns.row(post_i);

        for (arr_i, &conn) in conns.iter().enumerate() {
            let recent_spikes = &self.spike_history[conn as usize];

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

    //fn stdp(&self, pre_ms: f32, post_ms: f32) -> f32 {
    //    let d_syn = post_ms - pre_ms;

    //    if d_syn > 0.0 {
    //        self.a_plus * (E * (d_syn / self.tau_pre))
    //    } else {
    //        self.a_minus * (E * (d_syn / self.tau_pre))
    //    }
    //}

    pub fn run(&mut self, T: f32, input: Array2<f32>, logger: &mut Logger) -> ModelResponse {
        let dt: f32 = 0.1;
        let steps = (T / dt) as usize;

        let mut u_values = Array2::zeros((steps, self.n_neurons));
        let mut v_values = Array2::zeros((steps, self.n_neurons));
        let mut I_values = Array2::zeros((steps, self.n_neurons));

        let mut t_values = Array2::zeros((steps, self.n_neurons));

        let t_start = 50.0;
        let t_end = T;
        let I_baseline: f32 = 20.0;
        let log_interval = 1;

        let mut rng = rand::rng();
        for t in 0..steps {
            for i in 0..self.n_neurons {
                let mut I = input[[t, i]] + I_baseline;
                //println!("{}", logger.spikes_data);
                if t != 0 {
                    for (j, &conn_n_idx) in self.conns.row(i).iter().enumerate() {
                        let weight = self.weights[[i, j as usize]];
                        let pre_spike = logger.spikes_data[[t - 1, conn_n_idx as usize]] as f32;
                        let v_incoming = pre_spike * weight;

                        //println!("v inc: {} | {} | {}", v_incoming, weight, pre_spike);
                        I += v_incoming;
                    }
                }

                //let mut I = rng.random::<f32>() * 20.0;
                let t_ms = t as f32 * dt;

                //if t_ms >= t_start && t_ms <= t_end {
                //    I += I_baseline;
                //}

                if self.membranes[i] > self.threshold {
                    self.membranes[i] = self.c; // Reset
                    self.u[i] += self.d;
                    logger.spikes_data[[t, i]] = 1;

                    self.spike_history[i].push(t_ms);

                    while !self.spike_history[i].is_empty()
                        && t_ms - self.spike_history[i][0] > 50.0
                    {
                        self.spike_history[i].remove(0);
                    }
                    self.apply_stdp(i, t_ms);
                }

                let v = self.membranes[i];
                let u = self.u[i];
                self.membranes[i] += dt * 0.5 * (0.04_f32 * v.pow(2) + 5.0 * v + 140_f32 - u + I);
                self.u[i] += dt * self.a * (self.b * v - u);

                u_values[[t, i]] = u;
                v_values[[t, i]] = v;
                I_values[[t, i]] = I;
                t_values[[t, i]] = t_ms;
                if t != 0 {
                    if t % log_interval == 0 {
                        for c in 0..self.n_conns {
                            let conn_idx = c;
                            let global_col = i * self.n_conns + conn_idx;
                            let row = t / log_interval;

                            logger.weights[[row, global_col]] = self.weights[[i, conn_idx]];
                        }
                    }
                }
            }
        }

        ModelResponse {
            u_values,
            v_values,
            I_values,
            t_values,
            threshold: self.threshold,
        }
    }

    pub fn from_builder(builder: IzhikevichBuilder) -> Self {
        Self {
            n_neurons: builder.n_neurons,
            n_conns: builder.n_conns,
            threshold: builder.threshold,
            a: builder.a,
            b: builder.b,
            c: builder.c,
            d: builder.d,
            spike_history: vec![vec![0.0; 250]; builder.n_neurons],
            //spikes: Array2::zeros((builder.T, builder.n_neurons)),

            // STDP
            a_plus: builder.a_plus,
            a_minus: builder.a_minus,
            tau_plus: builder.tau_plus,
            tau_minus: builder.tau_minus,

            // Existing fields
            membranes: Array1::zeros(builder.n_neurons),
            u: Array1::zeros(builder.n_neurons),
            weights: Array2::random(
                (builder.n_neurons, builder.n_conns),
                Uniform::new(builder.min_weight, builder.max_weight).unwrap(),
            ),
            conns: Array2::random(
                (builder.n_neurons, builder.n_conns),
                Uniform::new(0, builder.n_neurons as u32).unwrap(),
            ),
            max_weight: builder.max_weight,
            min_weight: builder.min_weight,
        }
    }
}

pub struct IzhikevichBuilder {
    n_neurons: usize,
    n_conns: usize,
    threshold: f32,
    a: f32, // Time scale of the recovery variable. Smaller -> slower recovery
    b: f32, // Sensitivty of the recovery variable. Larger -> stronger coupling
    c: f32, // Reset voltage of membrane
    d: f32, // Increment of the recovery variable after a spike.

    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
    max_weight: f32,
    min_weight: f32,

    T: usize,
}

impl Default for IzhikevichBuilder {
    fn default() -> Self {
        Self {
            n_neurons: 1,
            n_conns: 1,
            threshold: 40.0,
            a: 0.01,
            b: 0.2,
            T: 100,
            c: -65.0,
            d: 8.0,
            a_plus: 0.01,
            a_minus: 0.02,
            tau_plus: 20.0,
            tau_minus: 20.0,
            min_weight: -1.0,
            max_weight: 1.0,
        }
    }
}

impl IzhikevichBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_neurons(mut self, n_neurons: usize) -> Self {
        self.n_neurons = n_neurons;
        self
    }
    pub fn n_conns(mut self, n_conns: usize) -> Self {
        self.n_conns = n_conns;
        self
    }

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

    pub fn T(mut self, T: usize) -> Self {
        self.T = T;
        self
    }

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
    pub fn max_weight(mut self, weight: f32) -> Self {
        self.max_weight = weight;
        self
    }
    pub fn min_weight(mut self, weight: f32) -> Self {
        self.min_weight = weight;
        self
    }

    pub fn build(self) -> Izhikevich {
        Izhikevich::from_builder(self)
    }
}
