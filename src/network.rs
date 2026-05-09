use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub struct Network {
    pub starting_conns: u32,
    pub n_neurons: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub base_threshold: f32,
    pub membranes: Array1<f32>,
    pub conns: Array2<u32>,   // Each neuron has conns
    pub n_conns: Array1<u32>, // Each neuron has number of conns
    pub weights: Array2<f32>,
    pub refactory: Array1<u32>,
    pub pre_spikes: Array1<u32>,
    pub post_spikes: Array1<u32>,
    pub pre_trace: Array1<f32>,
    pub post_trace: Array1<f32>,
    pub beta: f32,
    pub T: usize,

    pub output_neurons: Array1<u32>, //pub output_neurons: Array2<u32>,
}

impl Network {
    pub fn new(
        n_neurons: usize,
        starting_conns: u32,
        input_dim: usize,
        output_dim: usize,
        base_threshold: f32,
        beta: f32,
        w_min: f32,
        w_max: f32,
        T: usize,
    ) -> Self {
        Self {
            n_conns: Array1::from_elem(n_neurons, starting_conns),
            n_neurons,
            input_dim,
            T,
            output_neurons: Array1::zeros(output_dim),
            output_dim: input_dim,
            beta,
            starting_conns,
            base_threshold,
            membranes: Array1::zeros(n_neurons),
            refactory: Array1::zeros(n_neurons),
            pre_trace: Array1::zeros(n_neurons),
            post_trace: Array1::zeros(n_neurons),
            pre_spikes: Array1::zeros(n_neurons),
            post_spikes: Array1::zeros(n_neurons),
            weights: Array2::random(
                (n_neurons, starting_conns as usize),
                Uniform::new(w_min, w_max).unwrap(),
            ),
            conns: Array2::random(
                (n_neurons, starting_conns as usize),
                Uniform::new(0, n_neurons as u32).unwrap(),
            ),
        }
    }

    pub fn run(&mut self, input: &Array2<u32>, output_train: &mut Array2<u32>, timesteps: u32) {
        for t in 0..timesteps {
            let input = input.row(t as usize);
            self.step(&input, output_train, t);
        }
    }

    fn step(&mut self, input: &ArrayView1<u32>, output_train: &mut Array2<u32>, t: u32) {
        self.inject_input(input); // Set input neurons to input train at t
        self.accumulate(); // Sum all pre_spikes to connected
        self.threshold(); // Check threshold, set post_spikes and go to refactory
        self.push_output(output_train, t); // Push output neurons to output train
    }

    fn inject_input(&mut self, input: &ArrayView1<u32>) {
        assert!(
            input.len() == self.input_dim,
            "Parsed input is not equal to input_dim."
        );

        for i in 0..self.input_dim {
            self.pre_spikes[i] = input[i];
        }
    }

    fn accumulate(&mut self) {
        for i in 0..self.n_neurons {
            let spike = self.pre_spikes[i] == 1;
            if spike {
                for c in 0..self.n_conns[i] {
                    let weight = self.weights[[i, c as usize]];
                    let conn_n_idx = self.conns[[i, c as usize]] as usize;

                    if self.refactory[conn_n_idx] == 0 {
                        self.membranes[conn_n_idx] += weight;
                    }
                }
            }

            self.membranes[i] *= self.beta;
        }
    }

    fn threshold(&mut self) {
        for i in 0..self.n_neurons {
            if self.membranes[i] > self.base_threshold {
                self.post_spikes[i] = 1;
                self.refactory[i] = 5;
            } else {
                self.post_spikes[i] = 0;
                self.refactory[i].saturating_sub(1);
            }
        }
    }

    fn push_output(&mut self, output_train: &mut Array2<u32>, t: u32) {
        for (neuron, &idx) in self.output_neurons.iter().enumerate() {
            output_train[[t as usize, idx as usize]] = self.post_spikes[neuron];
        }
    }
}
