use std::time::Duration;

pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(5),
            multiplier: 2.0,
        }
    }
}

pub async fn with_retry<F, Fut, T, E, C>(
    config: &RetryConfig,
    mut operation: F,
    mut should_retry: C,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    C: FnMut(&E, u32) -> bool,
{
    let mut attempt = 0u32;
    let mut delay = config.initial_delay;

    loop {
        match operation().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                attempt += 1;
                if attempt >= config.max_attempts || !should_retry(&e, attempt) {
                    return Err(e);
                }
                tokio::time::sleep(delay).await;
                delay = std::cmp::min(
                    Duration::from_millis((delay.as_millis() as f64 * config.multiplier) as u64),
                    config.max_delay,
                );
            }
        }
    }
}
