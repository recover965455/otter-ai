use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll, Waker};

use futures::Stream;
use parking_lot::Mutex;

struct EventStreamInner<T, R> {
    queue: VecDeque<T>,
    wakers: VecDeque<Waker>,
    done: bool,
    result: Option<R>,
    result_ready: bool,
    result_wakers: VecDeque<Waker>,
}

pub struct EventStream<T, R> {
    inner: Arc<Mutex<EventStreamInner<T, R>>>,
    is_complete: Arc<dyn Fn(&T) -> bool + Send + Sync>,
    extract_result: Arc<dyn Fn(&T) -> R + Send + Sync>,
}

impl<T, R> EventStream<T, R>
where
    T: Clone + Send + 'static,
    R: Clone + Send + 'static,
{
    pub fn new<F1, F2>(is_complete: F1, extract_result: F2) -> Self
    where
        F1: Fn(&T) -> bool + Send + Sync + 'static,
        F2: Fn(&T) -> R + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(Mutex::new(EventStreamInner {
                queue: VecDeque::new(),
                wakers: VecDeque::new(),
                done: false,
                result: None,
                result_ready: false,
                result_wakers: VecDeque::new(),
            })),
            is_complete: Arc::new(is_complete),
            extract_result: Arc::new(extract_result),
        }
    }

    pub fn push(&self, event: T) {
        let mut inner = self.inner.lock();
        if inner.done {
            return;
        }

        if (self.is_complete)(&event) {
            inner.done = true;
            inner.result = Some((self.extract_result)(&event));
            inner.result_ready = true;
            for waker in inner.result_wakers.drain(..) {
                waker.wake();
            }
        }

        if let Some(waker) = inner.wakers.pop_front() {
            drop(inner);
            waker.wake();
            self.inner.lock().queue.push_back(event);
        } else {
            inner.queue.push_back(event);
        }
    }

    pub fn end(&self, result: Option<R>) {
        let mut inner = self.inner.lock();
        inner.done = true;
        if let Some(r) = result {
            inner.result = Some(r);
            inner.result_ready = true;
            let result_wakers: Vec<Waker> = inner.result_wakers.drain(..).collect();
            drop(inner);
            for waker in result_wakers {
                waker.wake();
            }
            inner = self.inner.lock();
        }
        let wakers: Vec<Waker> = inner.wakers.drain(..).collect();
        drop(inner);
        for waker in wakers {
            waker.wake();
        }
    }

    pub fn result_future(&self) -> impl std::future::Future<Output = R> + Send + 'static {
        let inner = self.inner.clone();
        async move {
            futures::future::poll_fn(|cx| {
                let mut guard = inner.lock();
                if guard.result_ready {
                    Poll::Ready(guard.result.clone().expect("result_ready without result"))
                } else {
                    guard.result_wakers.push_back(cx.waker().clone());
                    Poll::Pending
                }
            })
            .await
        }
    }
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            is_complete: self.is_complete.clone(),
            extract_result: self.extract_result.clone(),
        }
    }
}

impl<T, R> Stream for EventStream<T, R>
where
    T: Clone + Send + 'static,
    R: Send + 'static,
{
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let mut guard = self.inner.lock();
        if let Some(item) = guard.queue.pop_front() {
            Poll::Ready(Some(item))
        } else if guard.done {
            Poll::Ready(None)
        } else {
            guard.wakers.push_back(cx.waker().clone());
            Poll::Pending
        }
    }
}

pub type AssistantMessageEventStream =
    EventStream<crate::types::AssistantMessageEvent, crate::types::AssistantMessage>;

pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
    use crate::types::{AssistantMessage, AssistantMessageEvent, Message};

    fn is_complete_evt(event: &AssistantMessageEvent) -> bool {
        matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        )
    }

    fn extract_result_evt(event: &AssistantMessageEvent) -> AssistantMessage {
        match event {
            AssistantMessageEvent::Done { message, .. } => message.clone(),
            AssistantMessageEvent::Error { error, reason } => {
                let mut m = Message::assistant_default("unknown".into(), "unknown".into());
                if let Message::Assistant {
                    ref mut stop_reason,
                    ref mut error_message,
                    ..
                } = m
                {
                    *stop_reason = Some(reason.clone());
                    *error_message = Some(error.clone());
                }
                m
            }
            _ => panic!("unexpected event type for final result"),
        }
    }

    EventStream::new(is_complete_evt, extract_result_evt)
}
