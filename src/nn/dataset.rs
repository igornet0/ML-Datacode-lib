// Dataset structure for working with tables as ML datasets

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, BufReader};

use ndarray::Axis;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};

use crate::tensor::Tensor;
use crate::vm_value::Value;

/// Result of [`Dataset::split`].
#[derive(Debug, Clone)]
pub struct SplitResult {
    pub train: Dataset,
    pub test: Dataset,
    pub train_indices: Option<Vec<usize>>,
    pub test_indices: Option<Vec<usize>>,
}

/// `test_size`: fraction in (0, 1) or absolute test count in \[1, n − 1\] (whole number).
fn parse_test_size(ts: f64, n: usize) -> Result<usize, String> {
    if ts > 0.0 && ts < 1.0 {
        let tc = (n as f64 * ts).round() as usize;
        if tc == 0 || tc >= n {
            return Err(format!(
                "dataset split: invalid test size (test_count={}, n={})",
                tc, n
            ));
        }
        Ok(tc)
    } else if ts >= 1.0 && ts == ts.floor() {
        let tc = ts as usize;
        if tc == 0 || tc >= n {
            return Err(format!(
                "dataset split: invalid test size (test_count={}, n={})",
                tc, n
            ));
        }
        Ok(tc)
    } else {
        Err(
            "dataset split: test_size must be a fraction in (0,1) or an integer count in [1, n-1]"
                .to_string(),
        )
    }
}

/// `train_size`: fraction in (0, 1) or absolute train count in \[1, n − 1\] (whole number).
fn parse_train_size(tr: f64, n: usize) -> Result<usize, String> {
    if tr > 0.0 && tr < 1.0 {
        let train_n = (n as f64 * tr).round() as usize;
        if train_n == 0 || train_n >= n {
            return Err(format!(
                "dataset split: invalid train size (train_count={}, n={})",
                train_n, n
            ));
        }
        Ok(train_n)
    } else if tr >= 1.0 && tr == tr.floor() {
        let train_n = tr as usize;
        if train_n == 0 || train_n >= n {
            return Err(format!(
                "dataset split: invalid train size (train_count={}, n={})",
                train_n, n
            ));
        }
        Ok(train_n)
    } else {
        Err(
            "dataset split: train_size must be a fraction in (0,1) or an integer count in [1, n-1]"
                .to_string(),
        )
    }
}

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

    fn subset_by_indices(&self, idxs: &[usize]) -> Result<Dataset, String> {
        let features = self.features.take_rows(idxs)?;
        let targets = self.targets.take_rows(idxs)?;
        Ok(Dataset {
            features,
            targets,
            feature_names: self.feature_names.clone(),
            target_names: self.target_names.clone(),
        })
    }

    /// Append all rows from `other` in place. Feature/target column names and trailing shapes must match.
    pub fn concat_in_place(&mut self, other: &Dataset) -> Result<(), String> {
        if other.batch_size() == 0 {
            return Ok(());
        }
        if self.feature_names != other.feature_names || self.target_names != other.target_names {
            return Err(
                "dataset.concat: feature_names and target_names must match both datasets".to_string(),
            );
        }
        self.features = Tensor::concat_axis0_many(&[&self.features, other.features()])?;
        self.targets = Tensor::concat_axis0_many(&[&self.targets, other.targets()])?;
        Ok(())
    }

    /// Append rows from feature/target tensors (same layout as [`Dataset::from_tensors`]).
    pub fn push_tensor_rows(&mut self, mut features: Tensor, mut targets: Tensor) -> Result<(), String> {
        if features.shape.is_empty() || targets.shape.is_empty() {
            return Err("push_data: features and targets must be non-empty tensors".to_string());
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
                "push_data: batch size mismatch: features rows {} vs targets rows {}",
                features.shape[0], targets.shape[0]
            ));
        }
        if features.shape[0] == 0 {
            return Ok(());
        }
        if self.num_features() != features.shape[1] || self.num_targets() != targets.shape[1] {
            return Err(format!(
                "push_data: expected feature dim {} and target dim {}, got {} and {}",
                self.num_features(),
                self.num_targets(),
                features.shape[1],
                targets.shape[1]
            ));
        }
        self.features = self.features.concat_axis0(&features)?;
        self.targets = self.targets.concat_axis0(&targets)?;
        Ok(())
    }

    /// Split into train / test subsets.
    ///
    /// - `test_size` / `train_size`: at most one of them; omit both for default 20% test.
    ///   Each can be a **fraction** in (0, 1) or a **whole number** of samples (≥ 1), matching
    ///   sklearn-style semantics (float fraction vs int count).
    /// - With `shuffle == false`, row order is preserved: **train** rows come first, **test** rows
    ///   last (like `train_test_split(..., shuffle=False)`).
    pub fn split(
        &self,
        test_size: Option<f64>,
        train_size: Option<f64>,
        shuffle: bool,
        random_state: Option<u64>,
        stratify: bool,
        return_indices: bool,
    ) -> Result<SplitResult, String> {
        let n = self.features.shape[0];
        if self.targets.shape[0] != n {
            return Err(format!(
                "features/targets batch mismatch: {} vs {}",
                n,
                self.targets.shape[0]
            ));
        }
        if n == 0 {
            return Err("dataset split: empty dataset".to_string());
        }
        if test_size.is_some() && train_size.is_some() {
            return Err("dataset split: specify only one of test_size and train_size".to_string());
        }

        let test_count = if let Some(tr) = train_size {
            let train_n = parse_train_size(tr, n)?;
            n - train_n
        } else if let Some(ts) = test_size {
            parse_test_size(ts, n)?
        } else {
            let tc = (n as f64 * 0.2).round() as usize;
            if tc == 0 || tc >= n {
                return Err(format!(
                    "dataset split: invalid default test size (test_count={}, n={})",
                    tc, n
                ));
            }
            tc
        };

        if test_count == 0 || test_count >= n {
            return Err(format!(
                "dataset split: invalid test size (test_count={}, n={})",
                test_count, n
            ));
        }

        let (test_idx, train_idx) = if stratify {
            self.stratified_split_indices(n, test_count, shuffle, random_state)?
        } else {
            let mut indices: Vec<usize> = (0..n).collect();
            if shuffle {
                if let Some(seed) = random_state {
                    let mut rng = StdRng::seed_from_u64(seed);
                    indices.shuffle(&mut rng);
                } else {
                    indices.shuffle(&mut rand::thread_rng());
                }
            }
            // Train first, test last (sklearn order when shuffle is false).
            let train_n = n - test_count;
            let train_idx = indices[..train_n].to_vec();
            let test_idx = indices[train_n..].to_vec();
            (test_idx, train_idx)
        };

        let train_ds = self.subset_by_indices(&train_idx)?;
        let test_ds = self.subset_by_indices(&test_idx)?;
        Ok(SplitResult {
            train: train_ds,
            test: test_ds,
            train_indices: if return_indices {
                Some(train_idx)
            } else {
                None
            },
            test_indices: if return_indices {
                Some(test_idx)
            } else {
                None
            },
        })
    }

    fn stratified_split_indices(
        &self,
        n: usize,
        test_count: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<(Vec<usize>, Vec<usize>), String> {
        let t = self.targets.data();
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let row = t.index_axis(Axis(0), i);
            let v = row.iter().next().copied().unwrap_or(0.0);
            groups.entry(v.to_bits()).or_default().push(i);
        }
        if groups.is_empty() {
            return Err("dataset split: stratify requires targets".to_string());
        }

        let mut class_keys: Vec<u32> = groups.keys().copied().collect();
        class_keys.sort_unstable();
        let n_per_class: Vec<usize> = class_keys
            .iter()
            .map(|k| groups.get(k).map(|v| v.len()).unwrap_or(0))
            .collect();

        let test_per_class = allocate_stratify_test_counts(&n_per_class, n, test_count);

        let mut test_idx = Vec::with_capacity(test_count);
        let mut train_idx = Vec::with_capacity(n - test_count);

        if let Some(seed) = random_state {
            let mut rng = StdRng::seed_from_u64(seed);
            for (ki, &key) in class_keys.iter().enumerate() {
                let idxs = groups.get_mut(&key).unwrap();
                if shuffle {
                    idxs.shuffle(&mut rng);
                }
                let take = test_per_class[ki];
                if take > idxs.len() {
                    return Err("dataset split: stratify allocation exceeds class size".to_string());
                }
                test_idx.extend_from_slice(&idxs[..take]);
                train_idx.extend_from_slice(&idxs[take..]);
            }
        } else {
            let mut rng = rand::thread_rng();
            for (ki, &key) in class_keys.iter().enumerate() {
                let idxs = groups.get_mut(&key).unwrap();
                if shuffle {
                    idxs.shuffle(&mut rng);
                }
                let take = test_per_class[ki];
                if take > idxs.len() {
                    return Err("dataset split: stratify allocation exceeds class size".to_string());
                }
                test_idx.extend_from_slice(&idxs[..take]);
                train_idx.extend_from_slice(&idxs[take..]);
            }
        }

        Ok((test_idx, train_idx))
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
                feature_batch_data.extend_from_slice(&self.features.as_slice()[start_idx..end_idx]);
            }
            
            let feature_batch = Tensor::new(feature_batch_data, vec![batch_size_actual, num_features])?;

            // OPTIMIZATION: Pre-allocate with exact capacity needed
            let mut target_batch_data = Vec::with_capacity(batch_size_actual * num_targets);
            
            // OPTIMIZATION: More efficient data extraction
            for &idx in chunk {
                let start_idx = idx * num_targets;
                let end_idx = start_idx + num_targets;
                target_batch_data.extend_from_slice(&self.targets.as_slice()[start_idx..end_idx]);
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

    const CIFAR_RECORD: usize = 3073;
    const CIFAR_DIM: usize = 3072;

    fn cifar_feature_names() -> Vec<String> {
        (0..Self::CIFAR_DIM).map(|i| format!("x{}", i)).collect()
    }

    /// Один batch-файл CIFAR-10/100: записи по 3073 байта (метка + 3072 RGB).
    pub fn from_cifar_bin_file(path: &str) -> Result<Self, String> {
        let mut file = File::open(path)
            .map_err(|e| format!("Failed to open CIFAR bin {}: {}", path, e))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read CIFAR bin {}: {}", path, e))?;
        if buf.is_empty() {
            return Err(format!("Empty CIFAR file {}", path));
        }
        if buf.len() % Self::CIFAR_RECORD != 0 {
            return Err(format!(
                "CIFAR file {}: size {} not divisible by {}",
                path,
                buf.len(),
                Self::CIFAR_RECORD
            ));
        }
        let n = buf.len() / Self::CIFAR_RECORD;
        let mut feature_data = Vec::with_capacity(n * Self::CIFAR_DIM);
        let mut target_data = Vec::with_capacity(n);
        for row in 0..n {
            let off = row * Self::CIFAR_RECORD;
            target_data.push(buf[off] as f32);
            for i in 1..Self::CIFAR_RECORD {
                feature_data.push(buf[off + i] as f32 / 255.0);
            }
        }
        let features = Tensor::new(feature_data, vec![n, Self::CIFAR_DIM])?;
        let targets = Tensor::new(target_data, vec![n, 1])?;
        Ok(Dataset {
            features,
            targets,
            feature_names: Self::cifar_feature_names(),
            target_names: vec!["label".to_string()],
        })
    }

    /// Несколько batch-файлов CIFAR-10 (train): последовательное объединение строк.
    pub fn from_cifar10_bin_paths(paths: &[String]) -> Result<Self, String> {
        if paths.is_empty() {
            return Err("CIFAR-10: no bin paths".to_string());
        }
        let mut acc = Dataset::from_cifar_bin_file(&paths[0])?;
        for p in paths.iter().skip(1) {
            let next = Dataset::from_cifar_bin_file(p)?;
            acc.concat_in_place(&next)?;
        }
        Ok(acc)
    }
}

fn allocate_stratify_test_counts(n_per_class: &[usize], n: usize, test_count: usize) -> Vec<usize> {
    let k = n_per_class.len();
    if k == 0 {
        return Vec::new();
    }
    let mut q: Vec<usize> = n_per_class
        .iter()
        .map(|&nc| (nc.saturating_mul(test_count)) / n)
        .collect();
    let mut rem = test_count.saturating_sub(q.iter().sum());
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        let fa = (n_per_class[a] as f64) * (test_count as f64) / (n as f64) - q[a] as f64;
        let fb = (n_per_class[b] as f64) * (test_count as f64) / (n as f64) - q[b] as f64;
        fb.partial_cmp(&fa).unwrap()
    });
    for &i in &order {
        if rem == 0 {
            break;
        }
        if q[i] < n_per_class[i] {
            q[i] += 1;
            rem -= 1;
        }
    }
    if rem > 0 {
        for i in 0..k {
            if rem == 0 {
                break;
            }
            if q[i] < n_per_class[i] {
                q[i] += 1;
                rem -= 1;
            }
        }
    }
    let mut sumq: usize = q.iter().sum();
    while sumq > test_count {
        let mut best_i = None;
        let mut best_score = f64::INFINITY;
        for i in 0..k {
            if q[i] == 0 {
                continue;
            }
            let exact = (n_per_class[i] as f64) * (test_count as f64) / (n as f64);
            let score = q[i] as f64 - exact;
            if score < best_score {
                best_score = score;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            q[i] -= 1;
            sumq -= 1;
        } else {
            break;
        }
    }
    while sumq < test_count {
        let mut best_i = None;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..k {
            if q[i] >= n_per_class[i] {
                continue;
            }
            let exact = (n_per_class[i] as f64) * (test_count as f64) / (n as f64);
            let score = exact - q[i] as f64;
            if score > best_score {
                best_score = score;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            q[i] += 1;
            sumq += 1;
        } else {
            break;
        }
    }
    q
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

