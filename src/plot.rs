use ndarray::Array2;
use plotters::{
    prelude::*,
    style::full_palette::{ORANGE, PURPLE},
};
use rand::RngExt;

use crate::izhikevich::ModelResponse; // Adjust path if needed

pub fn plot_izhikievich(
    response: &ModelResponse,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(400);

    // === Top Plot: Membrane Potential (v) ===
    let mut v_chart = ChartBuilder::on(&upper)
        .caption(
            "Izhikevich Neuron - Membrane Potential (v)",
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..response.t_values[[response.t_values.nrows() - 1, 0]],
            -80f32..40f32,
        )?;

    v_chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("v (mV)")
        .draw()?;

    let colors = vec![RED, BLUE, GREEN, ORANGE, PURPLE];

    let mut rng = rand::rng();
    // Plot each neuron
    let n_neurons = response.v_values.ncols();
    for i in 0..n_neurons {
        //let color = colors[(i as f32 / 2.0).floor() as usize];
        let color = colors[rng.random_range(0..5)];
        //let color = Palette99::pick(i % 99).mix(0.9);
        let data: Vec<(f32, f32)> = (0..response.v_values.nrows())
            .map(|t| (response.t_values[[t, i]], response.v_values[[t, i]]))
            .collect();

        v_chart.draw_series(LineSeries::new(data, color.stroke_width(2)))?;
    }

    // === Bottom Plot: Recovery variable (u) ===
    let mut u_chart = ChartBuilder::on(&lower)
        .caption("Recovery Variable (u)", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..response.t_values[[response.t_values.nrows() - 1, 0]],
            -20f32..10f32,
        )?;

    u_chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("u")
        .draw()?;

    for i in 0..n_neurons {
        //let color = RED;
        let color = colors[rng.random_range(0..5)];
        let data: Vec<(f32, f32)> = (0..response.u_values.nrows())
            .map(|t| (response.t_values[[t, i]], response.u_values[[t, i]]))
            .collect();

        u_chart.draw_series(LineSeries::new(data, color.stroke_width(2)))?;
    }

    // Optional: Also save I (input current)
    println!("✅ Plots saved to: {}", output_path);
    Ok(())
}

pub fn plot_raster2(
    response: &ModelResponse,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut v_chart = ChartBuilder::on(&root)
        .caption(
            "Izhikevich Neuron - Membrane Potential (v)",
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..response.t_values[[response.t_values.nrows() - 1, 0]],
            -80f32..40f32,
        )?;

    let n_neurons = response.v_values.ncols();

    let threshold = 45.0;

    for i in 0..n_neurons {
        //let d: Vec<_> = (0..response.t_values.nrows()).map(|t| {
        //    response.v_values.iter().map(|&v| {
        //        if v > threshold {
        //            Circle::new((t, i), 2, BLACK.filled())
        //        } else {
        //            Circle::new((t, i), 0, BLACK.filled())
        //        }
        //    })
        //});
        //v_chart.draw_series(d)?;
    }

    Ok(())
}

pub fn plot_raster(
    spikes: &Array2<u32>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let n_neurons = spikes.ncols();
    let max_t = spikes.nrows() as f32 - 1.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Raster Plot - Spiking Activity", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..max_t, -0.5..(n_neurons as f32 - 0.5))?;

    chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("Neuron ID")
        .y_label_formatter(&|y| format!("{}", y.round() as i32))
        .draw()?;

    // Find spikes and plot them
    //for neuron in 0..n_neurons {
    //    let y = neuron as f32;

    //    for t in 0..response.v_values.nrows() {
    //        let current_v = response.v_values[[t, neuron]];
    //        let prev_v = if t > 0 {
    //            response.v_values[[t - 1, neuron]]
    //        } else {
    //            current_v
    //        };
    //        println!("{} {} {}", current_v, prev_v, response.threshold);

    //        // Detect spike: voltage crosses threshold from below
    //        if current_v > response.threshold && prev_v <= response.threshold {
    //            let time = response.t_values[[t, neuron]];

    //            chart.draw_series(vec![(Circle::new((time, y), 5, BLUE.filled()))])?;
    //        }
    //    }
    //}

    let threshold = 30;
    for neuron in 0..n_neurons {
        let y = neuron as f32;

        for t in 0..spikes.nrows() {
            let fired = spikes[[t, neuron]] == 1;

            // Robust spike detection for Izhikevich
            if fired {
                // Catch big overshoots
                chart.draw_series(std::iter::once(Circle::new((t as f32, y), 3, RED.filled())))?;
            }
        }
    }
    println!("✅ Raster plot saved to: {}", output_path);
    Ok(())
}
