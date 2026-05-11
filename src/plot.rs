use ndarray::Array2;
use plotters::{
    prelude::*,
    style::full_palette::{ORANGE, PURPLE},
};
use rand::RngExt;

use crate::{izhikevich::ModelResponse, layer::IzhikevichLayer}; // Adjust path if needed

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

pub fn plot_weights(
    data: Array2<f32>,
    output_path: &str,
    max_neurons: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let timesteps = data.nrows();
    let n_to_plot = std::cmp::min(max_neurons, data.ncols());

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Weight Evolution - First {} Neurons", n_to_plot),
            ("sans-serif", 30),
        )
        .margin(25)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0f32..timesteps as f32, -0.8f32..1.2f32)?;

    chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Synaptic Weight")
        .draw()?;

    for neuron in 0..n_to_plot {
        //let color = Palette99::pick(neuron).mix(0.9).stroke_width(2);

        let color = PURPLE;
        let series: Vec<(f32, f32)> = (0..timesteps)
            .map(|t| (t as f32, data[[t, neuron]]))
            .collect();

        chart.draw_series(LineSeries::new(series, color))?;
    }

    Ok(())
}

pub fn plot_weights2(
    weights_history: &Array2<f32>, // shape: (timesteps, n_neurons * n_conns_per_neuron) or (timesteps, n_neurons)
    n_conns_per_neuron: usize,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let timesteps = weights_history.nrows();
    let n_entries = weights_history.ncols();
    let n_neurons = n_entries / n_conns_per_neuron;

    let root = BitMapBackend::new(output_path, (1600, 1100)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top, bottom) = root.split_vertically(400);

    // ==================== TOP: Average Weight + Individual Lines ====================
    let mut avg_chart = ChartBuilder::on(&top)
        .caption("Synaptic Weight Evolution", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..timesteps as f32, -0.8f32..1.2f32)?;

    avg_chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Weight")
        .draw()?;

    // Compute average weight per timestep
    let mut avg_weights = vec![0.0; timesteps];
    for t in 0..timesteps {
        let sum: f32 = (0..n_entries).map(|j| weights_history[[t, j]]).sum();
        avg_weights[t] = sum / n_entries as f32;
    }

    // Plot average
    avg_chart
        .draw_series(LineSeries::new(
            (0..timesteps).map(|t| (t as f32, avg_weights[t])),
            BLUE.stroke_width(3),
        ))?
        .label("Average Weight")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Plot a few example neurons (first connection only)
    for neuron in 0..std::cmp::min(8, n_neurons) {
        let conn_idx = neuron * n_conns_per_neuron;
        let series: Vec<(f32, f32)> = (0..timesteps)
            .map(|t| (t as f32, weights_history[[t, conn_idx]]))
            .collect();

        avg_chart.draw_series(LineSeries::new(
            series,
            Palette99::pick(neuron).stroke_width(2),
        ))?;
    }

    // ==================== BOTTOM: Heatmap ====================
    let mut heatmap_chart = ChartBuilder::on(&bottom)
        .caption("Weight Heatmap (All Connections)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0..timesteps, 0..n_entries)?;

    heatmap_chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Connection Index (Neuron × Conn)")
        .draw()?;

    // Draw heatmap
    let max_abs = weights_history
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);

    for t in 0..timesteps {
        for j in 0..n_entries {
            let value = weights_history[[t, j]];
            let intensity = (value.abs() / (max_abs + 1e-6)).min(1.0);

            let color = if value >= 0.0 {
                RGBColor(255, (255.0 * intensity) as u8, (80.0 * intensity) as u8) // Red = positive
            } else {
                RGBColor((80.0 * intensity) as u8, (80.0 * intensity) as u8, 255) // Blue = negative
            };

            heatmap_chart.draw_series(std::iter::once(Rectangle::new(
                [(t, j), (t + 1, j + 1)],
                color.filled(),
            )))?;
        }
    }

    println!("✅ Weights visualization saved to: {}", output_path);
    Ok(())
}

use plotters::prelude::*;

pub fn plot_all_layers(
    layers: &[IzhikevichLayer],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_layers = layers.len();
    if n_layers == 0 {
        return Ok(());
    }

    let total_height = 280 * n_layers as u32 + 100;
    let root = SVGBackend::new(output_path, (1600, total_height)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((n_layers, 1));

    for (layer_idx, (layer, area)) in layers.iter().zip(areas.iter()).enumerate() {
        let logger = &layer.logger;

        let title = format!(
            "Layer {}: {} → {} neurons",
            layer_idx, layer.in_n, layer.out_n
        );

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 24))
            .margin(25)
            .x_label_area_size(45)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..logger.v_history.nrows() as f32, -85f32..45f32)?;

        chart
            .configure_mesh()
            .x_desc("Time Step")
            .y_desc("Membrane Potential (mV)")
            .draw()?;

        // Plot up to 8 neurons per layer
        let n_to_plot = std::cmp::min(8, layer.out_n);
        for n in 0..n_to_plot {
            let color = Palette99::pick(n + layer_idx * 8).stroke_width(2);

            let data: Vec<(f32, f32)> = (0..logger.v_history.nrows())
                .map(|t| (t as f32, logger.v_history[[t, n]]))
                .collect();

            chart.draw_series(LineSeries::new(data, color))?;
        }

        // Optional: Add spike markers
        // You can add raster-style dots if your logger has spike data
    }

    println!("✅ All layers plotted → {}", output_path);
    Ok(())
}

pub fn plot_all_layers_raster(
    layers: &[IzhikevichLayer],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_layers = layers.len();
    if n_layers == 0 {
        return Ok(());
    }

    // Calculate height based on number of layers and neurons
    let total_neurons: usize = layers.iter().map(|l| l.out_n).sum();
    let height = (total_neurons as u32 * 8).max(600) + 150;

    let root = SVGBackend::new(output_path, (1600, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((n_layers, 1));

    let mut global_neuron_offset = 0;

    for (layer_idx, (layer, area)) in layers.iter().zip(areas.iter()).enumerate() {
        let logger = &layer.logger;
        let n_neurons = layer.out_n;

        let title = format!(
            "Layer {} Raster | {} → {} neurons",
            layer_idx, layer.in_n, layer.out_n
        );

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 26))
            .margin(25)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(
                0f32..logger.v_history.nrows() as f32,
                (global_neuron_offset as f32 - 0.5)
                    ..(global_neuron_offset as f32 + n_neurons as f32 - 0.5),
            )?;

        chart
            .configure_mesh()
            .x_desc("Time Step")
            .y_desc("Neuron ID")
            .y_label_formatter(&|y| format!("{}", (*y as i32)))
            .draw()?;

        // Draw spikes for this layer
        for n in 0..n_neurons {
            let neuron_y = (global_neuron_offset + n) as f32;

            for t in 1..logger.v_history.nrows() {
                let v_prev = logger.v_history[[t - 1, n]];
                let v_curr = logger.v_history[[t, n]];

                // Detect spike
                if v_curr >= layer.threshold && v_prev < layer.threshold {
                    chart.draw_series(std::iter::once(Circle::new(
                        (t as f32, neuron_y),
                        2.5,
                        RED.filled(),
                    )))?;
                }
            }
        }

        global_neuron_offset += n_neurons;
    }

    println!("✅ Multi-layer Raster Plot saved to: {}", output_path);
    Ok(())
}
