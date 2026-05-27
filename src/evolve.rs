use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::{Distribution, num_traits};
use rand::distr::StandardUniform;
use rand::{RngExt, distr::uniform::SampleRange};

use crate::layer::{IzhikevichLayerBuilder, IzhikevichModel};

pub struct Reality {
    pub brains: Vec<Brain>,
    pub step: u32,
    //data: Array2<f32>,
    pub T: usize,
    pub running: bool,
}

pub struct Brain {
    life: f32,
    pub id: u32,
    pub model: IzhikevichModel,
}

impl Reality {
    pub fn new(T: usize) -> Self {
        Self {
            T,
            running: true,
            brains: Vec::new(),
            step: 0,
        }
    }

    pub fn tick(&mut self, input: ArrayView1<f32>, target: ArrayView1<f32>) {
        let mut accs: Vec<(u32, f32)> = Vec::new();

        //for (i, brain) in self.brains.iter_mut().enumerate() {
        //    let output = brain.model.run(self.T, input);
        //    let accuracy = check_acc(input, output);
        //    accs.push((i, accuracy));
        //}
        //accs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        //let max_survive = 10;

        //for j in max_survive..accs.len() {
        //    let (i, _) = accs[j];
        //    self.brains.remove(i);
        //}

        //for (i, acc) in accs.iter().take(max_survive) {
        //    let
        //}
        //

        //let mut perfect: HashMap<u32, f32> = HashMap::new();
        let mut perfect = Vec::new();
        self.brains.retain_mut(|brain| {
            let output = brain.model.run(self.T, input);
            let accuracy = check_acc(input, &output);
            //wrong += (accuracy * 9.).round() as u32;
            //println!("W: {}", wrong);
            let correct = check_xor(input, output.view(), brain.id);

            if correct == 9. {
                perfect.push(brain.id);
                //perfect.insert(brain.id, correct);
            }
            accs.push((brain.id, accuracy));
            brain.life += return_reward(9. - correct);

            //println!("Life: {}", brain.life);
            brain.life > 0.0
        });

        println!(
            "\nInput: {}\nperfect: {:?} {}",
            input,
            perfect,
            self.brains.len()
        );

        if perfect.len() > 1 {
            self.running = false;
        }
        //if wrong < 1 {
        //    self.running = false;
        //}

        accs.sort_by(|a, b| b.1.total_cmp(&a.1));
        //println!("acc 0: {} {}", accs[0].1, accs[8].1);

        let max_repr = 10;
        let mut n = 0;

        let mut offsprings = Vec::new();

        let n_brains = self.brains.len();
        while n < max_repr {
            if n >= n_brains {
                break;
            }
            let brain = &mut self.brains[n];
            // 1.07 exp
            //let thresh = 0.001 * self.step.pow(2) as f32 + 80.0;
            let thresh = 4. * 1.02.pow(2. * self.step as f32 + 40.) as f32 + 70.0;

            //println!("thresh: {}", thresh);
            if brain.life > thresh {
                let offspring = reproduce(&brain);
                offsprings.push(offspring);
            }
            n += 1;
        }

        //for brain in self.brains.iter_mut() {
        //    let f = accs.iter().find(|&b| b.1 == brain.id);
        //    let brain = &mut self.brains;
        //    // 1.07 exp
        //    //let thresh = 0.001 * self.step.pow(2) as f32 + 80.0;
        //    let thresh = 4. * 1.02.pow(2. * self.step as f32 + 40.) as f32 + 70.0;

        //    //println!("thresh: {}", thresh);
        //    if brain.life > thresh {
        //        let offspring = reproduce(&brain);
        //        offsprings.push(offspring);
        //    }
        //}

        //println!("Adding {} offsptriongs", offsprings.len());
        for offspring in offsprings {
            self.brains.push(offspring);
        }
        self.step += 1;
    }

    // Seed world with two brains
    pub fn adam_and_eve(&mut self) {
        let mother = Brain {
            model: build_brain(self.T),
            life: 100.,
            id: gen_id(),
        };

        let n = 10000;
        for i in 0..n {
            self.brains.push(reproduce(&mother));
        }
    }
}
pub fn check_acc(input: ArrayView1<f32>, output: &Array1<f32>) -> f32 {
    let mut wrong: f32 = 0.;
    for (i, &v) in input.iter().enumerate() {
        if v == 1. && output[i] == 1. || v == -1. && output[i] == -1. || v != 0. && output[i] == 0.
        {
            wrong += 1.;
        }
    }
    //if wrong < 3. {
    //    println!("Wrong: {}", wrong);
    //}

    let correct = 9. - wrong;

    correct / 9.
}

pub fn check_xor(input: ArrayView1<f32>, output: ArrayView1<f32>, id: u32) -> f32 {
    let mut correct: f32 = 0.0;

    for (i, &target) in input.iter().enumerate() {
        let out = output[i];

        let desired = if target < 0.0 { 1.0 } else { 0.0 };
        if out == desired {
            // approximately equal
            correct += 1.0;
        }
    }

    correct
}

//pub fn check_xor(input: ArrayView1<f32>, output: Array1<f32>, id: u32) -> f32 {
//    let mut wrong: f32 = 0.;
//    for (i, &v) in input.iter().enumerate() {
//        if v == 1. && output[i] == 1. || v == -1. && output[i] == -1. || v != 0. && output[i] == 0.
//        {
//            wrong += 1.;
//        }
//    }
//    //if wrong < 3. {
//    //    println!("{}: Wrong: {}", id, wrong,);
//    //}
//    //
//    wrong
//}

fn return_reward(wrong: f32) -> f32 {
    if wrong > 1. {
        -1.0 * wrong.pow(1.8) as f32
    } else {
        200.0
    }
}

fn build_brain(T: usize) -> IzhikevichModel {
    let mut model = IzhikevichModel::init();

    let input_layer = IzhikevichLayerBuilder::new()
        .in_n(9) // 3x3 grid
        .out_n(9)
        .n_conns(20)
        .threshold(40.0)
        .a(0.1)
        .b(0.7)
        .c(-35.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(0.0)
        .max_weight(3.0)
        .build();
    let hidden_layer = IzhikevichLayerBuilder::new()
        .in_n(9) // 3x3 grid
        .out_n(12)
        .n_conns(20)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(0.0)
        .max_weight(3.0)
        .build();
    let output_layer = IzhikevichLayerBuilder::new()
        .in_n(12) // 3x3 grid
        .out_n(9)
        .n_conns(20)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(0.0)
        .max_weight(3.0)
        .build();

    model.add_layer(input_layer);
    model.add_layer(hidden_layer);
    model.add_layer(output_layer);
    model
}

fn mutation<T, R>(parent_value: T, chance: f32, range: R) -> T
where
    T: num_traits::Num + rand::distr::uniform::SampleUniform,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
    rand::distr::StandardUniform: rand::distr::Distribution<f32>,
    //R: rand::distr::uniform::SampleUniform + SampleRange<T>,
    R: rand::distr::uniform::SampleRange<T>,
{
    let mut rng = rand::rng();

    let r: f32 = rng.random();

    if r < chance {
        rng.random_range(range)
    } else {
        parent_value
    }
}

fn get_weight(mother: &Brain) -> (f32, f32) {
    let mut min = mutation(mother.model.layers[0].min_weight, 0.4, 0.1..0.3);
    let mut max = mutation(mother.model.layers[0].max_weight, 0.4, min..3.3);
    while max <= min {
        min = mutation(mother.model.layers[0].min_weight, 0.4, 0.1..0.3);
        max = mutation(mother.model.layers[0].max_weight, 0.4, min..3.3);
    }
    (min, max)
}

fn reproduce(mother: &Brain) -> Brain {
    let mut model = IzhikevichModel::init();
    let max_T = 20;

    // Safe mutation helpers
    let out_first = mutation(mother.model.layers[0].out_n as u32, 0.15, 4..=32) as usize;
    let out_second = mutation(mother.model.layers[1].out_n as u32, 0.15, 4..=32) as usize;

    let (min_weight_1, max_weight_1) = get_weight(mother);
    let (min_weight_2, max_weight_2) = get_weight(mother);
    let (min_weight_3, max_weight_3) = get_weight(mother);

    //let min_weight_1 = mutation(mother.model.layers[0].min_weight,0.4, 0.1..0.3);
    //let max_weight_1 = mutation(mother.model.layers[0].max_weight,0.4, min_weight_1..0.3);

    //let min_weight_2 = mutation(mother.model.layers[0].min_weight,0.4, 0.1..0.3);
    //let max_weight_2 = mutation(mother.model.layers[0].max_weight,0.4, min_weight_2..0.3);

    //let min_weight_3 = mutation(mother.model.layers[0].min_weight,0.4, 0.1..0.3);
    //let max_weight_3 = mutation(mother.model.layers[0].max_weight,0.4, min_weight_3..0.3);

    let input_layer = IzhikevichLayerBuilder::new()
        .in_n(9)
        .out_n(out_first)
        .n_conns(mutation(mother.model.layers[0].n_conns as u32, 0.15, 8..=50) as usize)
        .threshold(mutation(mother.model.layers[0].threshold, 0.4, 40.0..70.))
        .a(mutation(mother.model.layers[0].a, 0.4, 0.001..0.3))
        .b(mutation(mother.model.layers[0].b, 0.4, 0.001..0.3)) // fixed: was using .a
        .c(mutation(mother.model.layers[1].c, 0.4, 0.0..75.))
        .d(mutation(mother.model.layers[1].d, 0.4, 0.0..12.0))
        .T(mutation(mother.model.layers[0].T as u32, 0.4, 10..=max_T) as usize)
        .a_plus(mutation(mother.model.layers[0].a_plus, 0.4, 0.001..0.2))
        .a_minus(mutation(mother.model.layers[0].a_minus, 0.4, 0.001..0.2))
        .tau_minus(mutation(mother.model.layers[0].tau_minus, 0.4, 0.001..20.5))
        .tau_plus(mutation(mother.model.layers[0].tau_plus, 0.4, 0.001..20.5))
        .min_weight(min_weight_1)
        .max_weight(max_weight_1)
        .build();

    let hidden_layer = IzhikevichLayerBuilder::new()
        .in_n(out_first)
        .out_n(out_second)
        .n_conns(mutation(mother.model.layers[1].n_conns as u32, 0.15, 8..=50) as usize)
        .threshold(mutation(mother.model.layers[1].threshold, 0.4, 40.0..70.))
        .a(mutation(mother.model.layers[1].a, 0.4, 0.001..0.3))
        .b(mutation(mother.model.layers[1].b, 0.4, 0.001..0.3)) // fixed: was using .a
        .c(mutation(mother.model.layers[1].c, 0.4, 0.0..75.))
        .d(mutation(mother.model.layers[1].d, 0.4, 0.0..12.0))
        .T(mutation(mother.model.layers[1].T as u32, 0.4, 10..=max_T) as usize)
        .a_plus(mutation(mother.model.layers[1].a_plus, 0.4, 0.001..0.2))
        .a_minus(mutation(mother.model.layers[1].a_minus, 0.4, 0.001..0.2))
        .tau_minus(mutation(mother.model.layers[1].tau_minus, 0.4, 0.001..20.5))
        .tau_plus(mutation(mother.model.layers[1].tau_plus, 0.4, 0.001..20.5))
        .min_weight(min_weight_2)
        .max_weight(max_weight_2)
        .build();

    let output_layer = IzhikevichLayerBuilder::new()
        .in_n(out_second)
        .out_n(9)
        .n_conns(mutation(mother.model.layers[2].n_conns as u32, 0.15, 8..=50) as usize)
        .threshold(mutation(mother.model.layers[2].threshold, 0.4, 40.0..70.))
        .a(mutation(mother.model.layers[2].a, 0.4, 0.001..0.3))
        .b(mutation(mother.model.layers[2].b, 0.4, 0.001..0.3)) // fixed: was using .a
        .c(mutation(mother.model.layers[1].c, 0.4, 0.0..75.))
        .d(mutation(mother.model.layers[1].d, 0.4, 0.0..12.0))
        .T(mutation(mother.model.layers[2].T as u32, 0.4, 10..=max_T) as usize)
        .a_plus(mutation(mother.model.layers[2].a_plus, 0.4, 0.001..0.2))
        .a_minus(mutation(mother.model.layers[2].a_minus, 0.4, 0.001..0.2))
        .tau_minus(mutation(mother.model.layers[2].tau_minus, 0.4, 0.001..20.5))
        .tau_plus(mutation(mother.model.layers[2].tau_plus, 0.4, 0.001..20.5))
        .min_weight(min_weight_3)
        .max_weight(max_weight_3)
        .build();

    model.add_layer(input_layer);
    model.add_layer(hidden_layer);
    model.add_layer(output_layer);

    Brain {
        model,
        life: 100.0,
        id: gen_id(),
    }
}

//fn reproduce3(mother: &Brain) -> Brain {
//    let mut model = IzhikevichModel::init();
//
//    let out_first = mutation(mother.model.layers[0].out_n as u32, 0.1) as usize;
//    let out_second = mutation(mother.model.layers[1].out_n as u32, 0.1) as usize;
//
//    let input_layer = IzhikevichLayerBuilder::new()
//        .in_n(9)
//        .out_n(out_first)
//        .n_conns(mutation(mother.model.layers[0].n_conns as u32, 0.1) as usize)
//        .threshold(mutation(mother.model.layers[0].threshold, 0.1))
//        .a(mutation(mother.model.layers[0].a, 0.1))
//        .b(mutation(mother.model.layers[0].a, 0.1))
//        .c(mutation(mother.model.layers[0].a, 0.1))
//        .d(mutation(mother.model.layers[0].a, 0.1))
//        .T(mutation(mother.model.layers[0].T as u32, 0.1) as usize)
//        .a_plus(mutation(mother.model.layers[0].a_plus, 0.1))
//        .a_minus(mutation(mother.model.layers[0].a_minus, 0.1))
//        .tau_minus(mutation(mother.model.layers[0].tau_minus, 0.1))
//        .tau_plus(mutation(mother.model.layers[0].tau_plus, 0.1))
//        .min_weight(mutation(mother.model.layers[0].min_weight, 0.1))
//        .max_weight(mutation(mother.model.layers[0].max_weight, 0.1))
//        .build();
//    let hidden_layer = IzhikevichLayerBuilder::new()
//        .in_n(out_first)
//        .out_n(out_second)
//        .n_conns(mutation(mother.model.layers[1].n_conns as u32, 0.1) as usize)
//        .threshold(mutation(mother.model.layers[1].threshold, 0.1))
//        .a(mutation(mother.model.layers[1].a, 0.1))
//        .b(mutation(mother.model.layers[1].a, 0.1))
//        .c(mutation(mother.model.layers[1].a, 0.1))
//        .d(mutation(mother.model.layers[1].a, 0.1))
//        .T(mutation(mother.model.layers[1].T as u32, 0.1) as usize)
//        .a_plus(mutation(mother.model.layers[1].a_plus, 0.1))
//        .a_minus(mutation(mother.model.layers[1].a_minus, 0.1))
//        .tau_minus(mutation(mother.model.layers[1].tau_minus, 0.1))
//        .tau_plus(mutation(mother.model.layers[1].tau_plus, 0.1))
//        .min_weight(mutation(mother.model.layers[1].min_weight, 0.1))
//        .max_weight(mutation(mother.model.layers[1].max_weight, 0.1))
//        .build();
//    let output_layer = IzhikevichLayerBuilder::new()
//        .in_n(out_second)
//        .out_n(9)
//        .n_conns(mutation(mother.model.layers[2].n_conns as u32, 0.1) as usize)
//        .threshold(mutation(mother.model.layers[2].threshold, 0.1))
//        .a(mutation(mother.model.layers[2].a, 0.1))
//        .b(mutation(mother.model.layers[2].a, 0.1))
//        .c(mutation(mother.model.layers[2].a, 0.1))
//        .d(mutation(mother.model.layers[2].a, 0.1))
//        .T(mutation(mother.model.layers[2].T as u32, 0.1) as usize)
//        .a_plus(mutation(mother.model.layers[2].a_plus, 0.1))
//        .a_minus(mutation(mother.model.layers[2].a_minus, 0.1))
//        .tau_minus(mutation(mother.model.layers[2].tau_minus, 0.1))
//        .tau_plus(mutation(mother.model.layers[2].tau_plus, 0.1))
//        .min_weight(mutation(mother.model.layers[2].min_weight, 0.1))
//        .max_weight(mutation(mother.model.layers[2].max_weight, 0.1))
//        .build();
//
//    model.add_layer(input_layer);
//    model.add_layer(hidden_layer);
//    model.add_layer(output_layer);
//    Brain { model, life: 100. }
//}

fn reproduce2(mother: &Brain) -> Brain {
    //let out_first = mutation(mother.model.layers[0].out_n as u32, 0.1) as usize;
    let T = 2;
    let mut model = IzhikevichModel::init();

    let input_layer = IzhikevichLayerBuilder::new()
        .in_n(9) // 3x3 grid
        .out_n(64)
        .n_conns(10)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-35.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();
    let hidden_layer = IzhikevichLayerBuilder::new()
        .in_n(9) // 3x3 grid
        .out_n(64)
        .n_conns(200)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();
    let output_layer = IzhikevichLayerBuilder::new()
        .in_n(64) // 3x3 grid
        .out_n(9)
        .n_conns(400)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();

    model.add_layer(input_layer);
    model.add_layer(hidden_layer);
    model.add_layer(output_layer);
    Brain {
        model,
        life: 100.,
        id: gen_id(),
    }
}

fn gen_id() -> u32 {
    let mut rng = rand::rng();

    rng.random::<u32>()
}
