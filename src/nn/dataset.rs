// Dataset structure for working with tables as ML datasets

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, BufReader};

use crate::tensor::Tensor;
use crate::vm_value::Value;

/// Dataset for ML operations
/// Converts table columns to tensors for training
#[derive(Debug, Clone)]
pub struct Dataset {
    features: Tensor,  // [batch_size, feature_count]
    targets: Tensor,   // [batch_size, target_count] or [batch_size, 1]
    feature_names: Vec<String>,
    target_names: Vec<String>,
}

impl Dataset {
    /// Decode `Value::Array([headers, rows])` from `AbiValue::Table` (see `plugin_abi_bridge`).
    pub fn parse_abi_table_from_value(v: &Value) -> Result<(Vec<String>, Vec<Vec<f32>>), String> {
        let outer = match v {
            Value::Array(a) => a.borrow(),
            _ => return Err("ml.dataset: table must be an array [headers, rows]".to_string()),
        };
        if outer.len() != 2 {
            return Err(format!(
                "ml.dataset: table array must have length 2 (headers, rows), got {}",
                outer.len()
            ));
        }
        let header_cells = match &outer[0] {
            Value::Array(h) => h.borrow(),
            _ => return Err("ml.dataset: table headers must be an array".to_string()),
        };
        let mut headers = Vec::with_capacity(header_cells.len());
        for hv in header_cells.iter() {
            match hv {
                Value::String(s) => headers.push(s.clone()),
                Value::Number(n) => headers.push(n.to_string()),
                _ => {
                    return Err(
                        "ml.dataset: header cells must be strings or numbers".to_string(),
                    )
                }
            }
        }
        let num_cols = headers.len();
        if num_cols == 0 {
            return Err("ml.dataset: table has no columns".to_string());
        }
        let row_vals = match &outer[1] {
            Value::Array(r) => r.borrow(),
            _ => return Err("ml.dataset: table rows must be an array".to_string()),
        };
        let mut rows = Vec::with_capacity(row_vals.len());
        for row_v in row_vals.iter() {
            let row_cells = match row_v {
                Value::Array(row) => row.borrow(),
                _ => {
                    return Err("ml.dataset: each row must be an array".to_string())
                }
            };
            if row_cells.len() != num_cols {
                return Err(format!(
                    "ml.dataset: row length {} does not match header count {}",
                    row_cells.len(),
                    num_cols
                ));
            }
            let mut rf = Vec::with_capacity(num_cols);
            for c in row_cells.iter() {
                match c {
                    Value::Number(n) => rf.push(*n as f32),
                    _ => {
                        return Err(
                            "ml.dataset: table cells must be numbers".to_string(),
                        )
                    }
                }
            }
            rows.push(rf);
        }
        Ok((headers, rows))
    }

    /// Build a dataset from column headers and row-major numeric rows (from ABI `Table`).
    pub fn from_abi_table(
        headers: &[String],
        rows: &[Vec<f32>],
        feature_columns: &[String],
        target_columns: &[String],
    ) -> Result<Self, String> {
        if feature_columns.is_empty() {
            return Err("At least one feature column is required".to_string());
        }
        if target_columns.is_empty() {
            return Err("At least one target column is required".to_string());
        }
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, h) in headers.iter().enumerate() {
            idx.insert(h.as_str(), i);
        }
        for col_name in feature_columns {
            if !idx.contains_key(col_name.as_str()) {
                return Err(format!("Feature column '{}' not found in table", col_name));
            }
        }
        for col_name in target_columns {
            if !idx.contains_key(col_name.as_str()) {
                return Err(format!("Target column '{}' not found in table", col_name));
            }
        }
        let num_rows = rows.len();
        if num_rows == 0 {
            return Err("Table is empty".to_string());
        }
        for (ri, row) in rows.iter().enumerate() {
            if row.len() != headers.len() {
                return Err(format!(
                    "Row {}: expected {} columns, got {}",
                    ri,
                    headers.len(),
                    row.len()
                ));
            }
        }

        let mut feature_data = Vec::with_capacity(num_rows * feature_columns.len());
        for col_name in feature_columns {
            let cix = *idx
                .get(col_name.as_str())
                .expect("column checked");
            for row in rows {
                feature_data.push(row[cix]);
            }
        }
        let num_features = feature_columns.len();
        let features = Tensor::new(feature_data, vec![num_rows, num_features])?;

        let mut target_data = Vec::with_capacity(num_rows * target_columns.len());
        for col_name in target_columns {
            let cix = *idx
                .get(col_name.as_str())
                .expect("column checked");
            for row in rows {
                target_data.push(row[cix]);
            }
        }
        let num_targets = target_columns.len();
        let targets = Tensor::new(target_data, vec![num_rows, num_targets])?;

        Ok(Dataset {
            features,
            targets,
            feature_names: feature_columns.to_vec(),
            target_names: target_columns.to_vec(),
        })
    }

    /// Build a dataset from feature/target tensors (e.g. when only tensors are available).
    pub fn from_tensors(mut features: Tensor, mut targets: Tensor) -> Result<Self, String> {
        if features.shape.is_empty() || targets.shape.is_empty() {
            return Err("features and targets must be non-empty tensors".to_string());
        }
        if features.shape.len() == 1 {
            let n = features.shape[0];
            features = features.reshape(vec![n, 1])?;
        }
        if targets.shape.len() == 1 {
            let n = targets.shape[0];
            targets = targets.reshape(vec![n, 1])?;
        }
        if features.shape[0] != targets.shape[0] {
            return Err(format!(
                "batch size mismatch: features rows {} vs targets rows {}",
                features.shape[0], targets.shape[0]
            ));
        }
        let nf = features.shape[1];
        let nt = targets.shape[1];
        let feature_names: Vec<String> = (0..nf).map(|i| format!("x{}", i)).collect();
        let target_names: Vec<String> = (0..nt).map(|i| format!("y{}", i)).collect();
        Ok(Dataset {
            features,
            targets,
            feature_names,
            target_names,
        })
    }

    /// Get features tensor
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// Get targets tensor
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get target names
    pub fn target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get batch size (number of samples)
    pub fn batch_size(&self) -> usize {
        self.features.shape[0]
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.features.shape[1]
    }

    /// Get number of targets
    pub fn num_targets(&self) -> usize {
        self.targets.shape[1]
    }

    /// Get batches from the dataset
    /// Returns vector of (features_batch, targets_batch) tuples
    pub fn batches(&self, batch_size: usize, shuffle: bool) -> Result<Vec<(Tensor, Tensor)>, String> {
        if batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        let num_samples = self.batch_size();
        if batch_size > num_samples {
            return Err(format!(
                "Batch size {} is larger than dataset size {}",
                batch_size, num_samples
            ));
        }

        // Create indices
        let mut indices: Vec<usize> = (0..num_samples).collect();

        // Shuffle if requested
        if shuffle {
            // Simple Fisher-Yates shuffle with fixed seed for reproducibility
            let mut seed: u64 = 42;
            for i in (1..num_samples).rev() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (seed % (i + 1) as u64) as usize;
                indices.swap(i, j);
            }
        }

        // OPTIMIZATION: Pre-allocate batches vector with known capacity
        let num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
        let mut batches = Vec::with_capacity(num_batches);

        let num_features = self.num_features();
        let num_targets = self.num_targets();

        // Split into batches
        for chunk in indices.chunks(batch_size) {
            let batch_size_actual = chunk.len();
            
            // OPTIMIZATION: Pre-allocate with exact capacity needed
            let mut feature_batch_data = Vec::with_capacity(batch_size_actual * num_features);
            
            // OPTIMIZATION: More efficient data extraction - iterate once and copy contiguous chunks
            for &idx in chunk {
                let start_idx = idx * num_features;
                let end_idx = start_idx + num_features;
                feature_batch_data.extend_from_slice(&self.features.data[start_idx..end_idx]);
            }
            
            let feature_batch = Tensor::new(feature_batch_data, vec![batch_size_actual, num_features])?;

            // OPTIMIZATION: Pre-allocate with exact capacity needed
            let mut target_batch_data = Vec::with_capacity(batch_size_actual * num_targets);
            
            // OPTIMIZATION: More efficient data extraction
            for &idx in chunk {
                let start_idx = idx * num_targets;
                let end_idx = start_idx + num_targets;
                target_batch_data.extend_from_slice(&self.targets.data[start_idx..end_idx]);
            }
            
            let target_batch = Tensor::new(target_batch_data, vec![batch_size_actual, num_targets])?;

            batches.push((feature_batch, target_batch));
        }

        Ok(batches)
    }

    /// Create a dataset from MNIST files
    /// 
    /// # Arguments
    /// * `images_path` - Path to MNIST images file (idx3-ubyte format)
    /// * `labels_path` - Path to MNIST labels file (idx1-ubyte format)
    /// 
    /// # Returns
    /// Dataset with features as normalized images [num_samples, 784] and targets as labels [num_samples, 1]
    pub fn from_mnist(images_path: &str, labels_path: &str) -> Result<Self, String> {
        let images = load_mnist_images(images_path)?;
        let labels = load_mnist_labels(labels_path)?;

        if images.len() != labels.len() {
            return Err(format!(
                "Mismatch: {} images but {} labels",
                images.len(),
                labels.len()
            ));
        }

        let num_samples = images.len();
        let image_size = 28 * 28; // 784 pixels per image

        // Flatten images and normalize to [0, 1]
        let mut feature_data = Vec::with_capacity(num_samples * image_size);
        for image in &images {
            for &pixel in image {
                feature_data.push(pixel as f32 / 255.0);
            }
        }

        // Convert labels to f32
        let mut target_data = Vec::with_capacity(num_samples);
        for &label in &labels {
            target_data.push(label as f32);
        }

        let features = Tensor::new(feature_data, vec![num_samples, image_size])?;
        let targets = Tensor::new(target_data, vec![num_samples, 1])?;

        Ok(Dataset {
            features,
            targets,
            feature_names: vec!["pixel".to_string(); image_size],
            target_names: vec!["label".to_string()],
        })
    }
}

/// Load MNIST images from IDX file format
/// 
/// # Arguments
/// * `path` - Path to the idx3-ubyte file
/// 
/// # Returns
/// Vector of images, each image is a Vec<u8> of 784 pixels (28x28)
pub fn load_mnist_images(path: &str) -> Result<Vec<Vec<u8>>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open images file {}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    // Read magic number (4 bytes, big-endian)
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)
        .map_err(|e| format!("Failed to read magic number: {}", e))?;
    let magic = u32::from_be_bytes(magic_bytes);
    
    if magic != 0x00000803 {
        return Err(format!("Invalid magic number: expected 0x00000803, got 0x{:08X}", magic));
    }

    // Read number of images (4 bytes, big-endian)
    let mut num_images_bytes = [0u8; 4];
    reader.read_exact(&mut num_images_bytes)
        .map_err(|e| format!("Failed to read number of images: {}", e))?;
    let num_images = u32::from_be_bytes(num_images_bytes) as usize;

    // Read number of rows (4 bytes, big-endian)
    let mut num_rows_bytes = [0u8; 4];
    reader.read_exact(&mut num_rows_bytes)
        .map_err(|e| format!("Failed to read number of rows: {}", e))?;
    let num_rows = u32::from_be_bytes(num_rows_bytes) as usize;

    // Read number of columns (4 bytes, big-endian)
    let mut num_cols_bytes = [0u8; 4];
    reader.read_exact(&mut num_cols_bytes)
        .map_err(|e| format!("Failed to read number of columns: {}", e))?;
    let num_cols = u32::from_be_bytes(num_cols_bytes) as usize;

    let image_size = num_rows * num_cols;

    // Read all image data
    let mut images = Vec::with_capacity(num_images);
    for _ in 0..num_images {
        let mut image = vec![0u8; image_size];
        reader.read_exact(&mut image)
            .map_err(|e| format!("Failed to read image data: {}", e))?;
        images.push(image);
    }

    Ok(images)
}

/// Load MNIST labels from IDX file format
/// 
/// # Arguments
/// * `path` - Path to the idx1-ubyte file
/// 
/// # Returns
/// Vector of labels (each label is 0-9)
pub fn load_mnist_labels(path: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open labels file {}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    // Read magic number (4 bytes, big-endian)
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)
        .map_err(|e| format!("Failed to read magic number: {}", e))?;
    let magic = u32::from_be_bytes(magic_bytes);
    
    if magic != 0x00000801 {
        return Err(format!("Invalid magic number: expected 0x00000801, got 0x{:08X}", magic));
    }

    // Read number of labels (4 bytes, big-endian)
    let mut num_labels_bytes = [0u8; 4];
    reader.read_exact(&mut num_labels_bytes)
        .map_err(|e| format!("Failed to read number of labels: {}", e))?;
    let num_labels = u32::from_be_bytes(num_labels_bytes) as usize;

    // Read all labels
    let mut labels = vec![0u8; num_labels];
    reader.read_exact(&mut labels)
        .map_err(|e| format!("Failed to read labels: {}", e))?;

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_dataset_from_abi_table() {
        let headers = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
        let rows = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let dataset = Dataset::from_abi_table(
            &headers,
            &rows,
            &["x1".to_string(), "x2".to_string()],
            &["y".to_string()],
        )
        .unwrap();

        assert_eq!(dataset.batch_size(), 2);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.num_targets(), 1);
        assert_eq!(dataset.features().shape, vec![2, 2]);
        assert_eq!(dataset.targets().shape, vec![2, 1]);
    }

    #[test]
    fn test_parse_abi_table_from_value() {
        let v = Value::Array(Rc::new(RefCell::new(vec![
            Value::Array(Rc::new(RefCell::new(vec![
                Value::String("x1".into()),
                Value::String("y".into()),
            ]))),
            Value::Array(Rc::new(RefCell::new(vec![Value::Array(Rc::new(
                RefCell::new(vec![Value::Number(1.0), Value::Number(3.0)]),
            ))]))),
        ])));
        let (h, r) = Dataset::parse_abi_table_from_value(&v).unwrap();
        assert_eq!(h, vec!["x1", "y"]);
        assert_eq!(r, vec![vec![1.0_f32, 3.0]]);
    }
}

