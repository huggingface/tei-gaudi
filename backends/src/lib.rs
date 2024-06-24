mod dtype;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use text_embeddings_backend_core::{Backend as CoreBackend, Predictions};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{instrument, Span};
use rand::Rng;
pub use crate::dtype::DType;
pub use text_embeddings_backend_core::{
    BackendError, Batch, Embedding, Embeddings, ModelType, Pool,
};

#[cfg(feature = "candle")]
use text_embeddings_backend_candle::CandleBackend;

#[cfg(feature = "python")]
use text_embeddings_backend_python::PythonBackend;

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread
    backend_sender: mpsc::UnboundedSender<BackendCommand>,
    /// Health status
    health_receiver: watch::Receiver<bool>,
    _backend_thread: Arc<BackendThread>,
    pub padded_model: bool,
    pub max_batch_size: Option<usize>,
    pub model_type: ModelType,
}

fn powers_of_two(max_value: u32) -> Vec<u32> {
    let mut result = Vec::new();
    let mut power: u32 = 1;

    while power <= max_value {
        result.push(power);
        power *= 2;
    }

    result
}

impl Backend {
    pub fn new(
        model_path: PathBuf,
        dtype: DType,
        model_type: ModelType,
        uds_path: String,
        otlp_endpoint: Option<String>,
    ) -> Result<Self, BackendError> {
        let (backend_sender, backend_receiver) = mpsc::unbounded_channel();

        let backend = init_backend(
            model_path,
            dtype,
            model_type.clone(),
            uds_path,
            otlp_endpoint,
        )?;
        let padded_model = backend.is_padded();
        let max_batch_size = backend.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);
        let _backend_thread =
            Arc::new(BackendThread::new(backend, backend_receiver, health_sender));

        Ok(Self {
            backend_sender,
            health_receiver,
            _backend_thread,
            padded_model,
            max_batch_size,
            model_type,
        })
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> Result<(), BackendError> {
        if *self.health_receiver.borrow() {
            // The backend is healthy. Only do a basic health check by calling the
            // the underlying health method.

            let (sender, receiver) = oneshot::channel();
            self.backend_sender
                .send(BackendCommand::Health(Span::current(), sender))
                .expect("No backend receiver. This is a bug.");
            receiver.await.expect(
                "Backend blocking task dropped the sender without sending a response. This is a bug.",
            )
        } else {
            // The backend is un-healthy or only just started. Do a more advanced health check
            // by calling the model forward on a test batch

            let batch = Batch {
                input_ids: vec![0],
                token_type_ids: vec![0],
                position_ids: vec![0],
                cumulative_seq_lengths: vec![0, 1],
                max_length: 1,
                pooled_indices: vec![0],
                raw_indices: vec![],
            };
            match &self.model_type {
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }
        }
    }

    #[instrument(skip(self))]
    pub async fn warmup(
        &self,
        mut max_input_length: u32,
        max_token: u32,
        max_bs: Option<usize>
    ) -> Result<(), BackendError> {
        let read_env_var = |key: &str, default: u32| -> u32 {
            env::var(key).ok().map_or(default, |value| value.parse::<u32>().unwrap())
        };
        let seq_bucket_size: u32 = read_env_var("PAD_SEQUENCE_TO_MULTIPLE_OF", 128);
        let max_warmup_length: u32 = read_env_var("MAX_WARMUP_SEQUENCE_LENGTH", 1024);

        let max_batch_size = match max_bs {
            Some(value) => value as u32,
            None => read_env_var("MAX_WARMUP_BATCH_SIZE", 8),
        };
        let mut batch_sizes: Vec<u32> = powers_of_two(max_batch_size);
        if let Some(&last) = batch_sizes.last() {
            if last < max_batch_size {
                batch_sizes.push(max_batch_size);
            }
        }
        if max_warmup_length > max_input_length {
            tracing::warn!("max_warmup_length exceeds model's max_input_length limit, will replace it");
        }
        max_input_length = std::cmp::min(max_input_length, max_warmup_length);
        let mut seq_lengths: Vec<u32> = (seq_bucket_size..max_input_length+1).step_by(seq_bucket_size as usize).collect();
        if let Some(&last) = seq_lengths.last() {
            if last < max_input_length {
                seq_lengths.push(max_input_length);
            }
        }

        let mut shapes: Vec<(u32, u32)> = Vec::with_capacity(batch_sizes.len() * seq_lengths.len());
        for batch_size in &batch_sizes {
            for seq_length in &seq_lengths {
                shapes.push((*batch_size, *seq_length));
            }
        }
        for shape in shapes.iter() {
            let batch = self.create_warmup_batch(*shape, max_token);
            match &self.model_type {
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }?;
            tracing::info!("finish warmup for batch: {}, length: {}", shape.0, shape.1);
        }
        Ok(())
    }

    #[instrument(skip_all)]
    pub fn create_warmup_batch(
        &self,
        shape: (u32, u32),
        max_token: u32,
    ) -> Batch {
        let (batch_size, length) = shape;
        let mut batched_input_ids = Vec::new();
        let mut batched_token_type_ids = Vec::new();
        let mut batched_position_ids = Vec::new();
        let mut cumulative_seq_lengths = Vec::with_capacity(batch_size as usize + 1);
        let mut pooled_indices = Vec::with_capacity(batch_size as usize);
        cumulative_seq_lengths.push(0);
        let input_ids: Vec<u32> = (0..length).map(|_| rand::thread_rng().gen_range(0..max_token)).collect();
        let token_type_ids: Vec<u32> = vec![0; length as usize];
        let position_ids: Vec<u32> = (0..length).collect();
        let mut current_length = 0;
        for batch_id in 0..batch_size {
            batched_input_ids.extend(input_ids.iter().cloned());
            batched_token_type_ids.extend(token_type_ids.iter().cloned());
            batched_position_ids.extend(position_ids.iter().cloned());
            current_length += input_ids.len();
            cumulative_seq_lengths.push(current_length as u32);
            pooled_indices.push(batch_id);
        }
        Batch {
            input_ids: batched_input_ids,
            token_type_ids: batched_token_type_ids,
            position_ids: batched_position_ids,
            cumulative_seq_lengths,
            max_length: length,
            pooled_indices,
            raw_indices: vec![],
        }
    }
    #[instrument(skip(self))]
    pub fn health_watcher(&self) -> watch::Receiver<bool> {
        self.health_receiver.clone()
    }

    #[instrument(skip_all)]
    pub async fn embed(&self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();
        self.backend_sender
            .send(BackendCommand::Embed(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Predictions, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Predict(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }
}

#[allow(unused)]
fn init_backend(
    model_path: PathBuf,
    dtype: DType,
    model_type: ModelType,
    uds_path: String,
    otlp_endpoint: Option<String>,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    if cfg!(feature = "candle") {
        #[cfg(feature = "candle")]
        return Ok(Box::new(CandleBackend::new(
            model_path,
            dtype.to_string(),
            model_type,
        )?));
    } else if cfg!(feature = "python") {
        #[cfg(feature = "python")]
        {
            return Ok(Box::new(
                std::thread::spawn(move || {
                    PythonBackend::new(
                        model_path.to_str().unwrap().to_string(),
                        dtype.to_string(),
                        model_type,
                        uds_path,
                        otlp_endpoint,
                    )
                })
                .join()
                .expect("Python Backend management thread failed")?,
            ));
        }
    }
    Err(BackendError::NoBackend)
}

#[derive(Debug)]
struct BackendThread(Option<JoinHandle<()>>);

impl BackendThread {
    fn new(
        backend: Box<dyn CoreBackend + Send>,
        mut backend_receiver: mpsc::UnboundedReceiver<BackendCommand>,
        health_sender: watch::Sender<bool>,
    ) -> Self {
        let handle = std::thread::spawn(move || {
            while let Some(cmd) = backend_receiver.blocking_recv() {
                let start = Instant::now();
                let mut healthy = false;
                match cmd {
                    BackendCommand::Health(span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.health().map(|_| healthy = true));
                    }
                    BackendCommand::Embed(batch, span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.embed(batch).map(|e| {
                            healthy = true;
                            (e, start.elapsed())
                        }));
                    }
                    BackendCommand::Predict(batch, span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.predict(batch).map(|e| {
                            healthy = true;
                            (e, start.elapsed())
                        }));
                    }
                };
                let _ = health_sender.send(healthy);
            }
        });
        Self(Some(handle))
    }
}

impl Drop for BackendThread {
    fn drop(&mut self) {
        self.0.take().unwrap().join().unwrap();
    }
}

enum BackendCommand {
    Health(Span, oneshot::Sender<Result<(), BackendError>>),
    Embed(
        Batch,
        Span,
        oneshot::Sender<Result<(Embeddings, Duration), BackendError>>,
    ),
    Predict(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<(Predictions, Duration), BackendError>>,
    ),
}
