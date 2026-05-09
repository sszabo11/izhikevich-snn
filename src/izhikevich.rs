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

pub struct Izhikevich {
    membranes: Array1<f32>,
    u: Array1<f32>,         // Recovery var for each neuron
    pub conns: Array2<u32>, // Each neuron has conns
    pub weights: Array2<f32>,

    pub spikes: Array2<u32>,

    n_neurons: usize,
    n_conns: usize,
    threshold: f32,
    a: f32, // Time scale of the recovery variable. Smaller -> slower recovery
    b: f32, // Sensitivty of the recovery variable. Larger -> stronger coupling
    c: f32, // Reset voltage of membrane
    d: f32, // Increment of the recovery variable after a spike.

    a_plus: f32,
    a_minus: f32,
    tau_pre: f32,
    tau_post: f32,
}

impl Izhikevich {
    pub fn new(
        n_neurons: usize,
        n_conns: usize,
        threshold: f32,
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
            threshold,
            n_conns,
            membranes: Array1::zeros(n_neurons),
            spikes: Array2::zeros((T * 10, n_neurons)),
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

    fn learn(&self, pre_ms: f32, post_ms: f32) -> f32 {
        let d_syn = post_ms - pre_ms;

        if d_syn > 0.0 {
            self.a_plus * (E * (d_syn / self.tau_pre))
        } else {
            self.a_minus * (E * (d_syn / self.tau_pre))
        }
    }

    pub fn run(&mut self, T: f32) -> ModelResponse {
        let dt: f32 = 0.1;
        let steps = (T / dt) as usize;

        let mut u_values = Array2::zeros((steps, self.n_neurons));
        let mut v_values = Array2::zeros((steps, self.n_neurons));
        let mut I_values = Array2::zeros((steps, self.n_neurons));

        let mut t_values = Array2::zeros((steps, self.n_neurons));

        let t_start = 50.0;
        let t_end = T;
        let I_baseline: f32 = 10.0;

        let mut rng = rand::rng();
        for t in 0..steps {
            for i in 0..self.n_neurons {
                let mut I = rng.random::<f32>() * 20.0;
                let t_ms = t as f32 * dt;

                if t_ms >= t_start && t_ms <= t_end {
                    I += I_baseline;
                }

                if self.membranes[i] > self.threshold {
                    self.membranes[i] = self.c; // Reset
                    self.u[i] += self.d;
                    self.spikes[[t, i]] = 1;
                }

                let v = self.membranes[i];
                let u = self.u[i];
                self.membranes[i] += dt * 0.5 * (0.04_f32 * v.pow(2) + 5.0 * v + 140_f32 - u + I);
                self.u[i] += dt * self.a * (self.b * v - u);

                u_values[[t, i]] = u;
                v_values[[t, i]] = v;
                I_values[[t, i]] = I;
                t_values[[t, i]] = t_ms;

                self.learn(pre_ms, post_ms)
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
}
