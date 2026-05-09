use anyhow::Result;
use snn_2::{
    izhikevich::Izhikevich,
    plot::{plot_izhikievich, plot_raster},
    plt::Plt,
};

fn main() -> Result<()> {
    let n_neurons = 20;
    let n_conns = 10;
    let threshold = 40.0;
    let a = 0.01; // Time scale of the recovery variable. Smaller -> slower recovery
    let b = 0.2; // Sensitivty of the recovery variable. Larger -> stronger coupling
    let c = -65.0; // Reset voltage of membrane                                       
    let d = 8.0; // Increment of the recovery variable after a spike.

    let T = 500;
    let mut model = Izhikevich::new(n_neurons, n_conns, threshold, a, b, c, d, T);

    let output = model.run(T as f32);

    //Plt::spikes("spikes.png", output)?;

    let rt = plot_izhikievich(&output, "charts/spikes3.png").unwrap();
    plot_raster(&model.spikes, "charts/raster2.png").unwrap();
    Ok(())
}
