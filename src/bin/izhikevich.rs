use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use snn_2::{
    izhikevich::{Izhikevich, IzhikevichBuilder},
    plot::{plot_izhikievich, plot_raster},
    plt::Plt,
};

fn main() -> Result<()> {
    //let n_conns = 10;
    //let threshold = 40.0;
    //let a = 0.01; // Time scale of the recovery variable. Smaller -> slower recovery
    //let b = 0.2; // Sensitivty of the recovery variable. Larger -> stronger coupling
    //let c = -65.0; // Reset voltage of membrane
    //let d = 8.0; // Increment of the recovery variable after a spike.

    let T = 500;
    //let mut model = Izhikevich::new(n_neurons, n_conns, threshold, 20.0, 20.0, a, b, c, d, T);

    let n_neurons = 20;
    let mut model = IzhikevichBuilder::new()
        .n_neurons(n_neurons)
        .n_conns(20)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .d(8.0)
        .a_plus(0.01)
        .a_minus(0.012)
        .T(T)
        .build();

    let steps = (T as f32 / 0.01) as usize;

    let input = Array2::random((steps, n_neurons), Uniform::new(0.0, 6.0).unwrap());

    let output = model.run(T as f32, input);

    //Plt::spikes("spikes.png", output)?;

    let rt = plot_izhikievich(&output, "charts/spikes3.png").unwrap();
    plot_raster(&model.spikes, "charts/raster2.png").unwrap();
    Ok(())
}
