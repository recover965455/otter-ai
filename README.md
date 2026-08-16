# otter-ai

[![crates.io](https://img.shields.io/crates/v/otter-ai.svg)](https://crates.io/crates/otter-ai)
[![docs.rs](https://docs.rs/otter-ai/badge.svg)](https://docs.rs/otter-ai)
[![license](https://img.shields.io/crates/l/otter-ai.svg)](#license)

> 统一的大语言模型 API SDK — 多提供商聚合、自动认证解析、Token 与成本追踪、上下文持久化。

`otter-ai` 是 TypeScript 包 [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi-ai) 的 Rust 重写版本，提供了一个统一的接口来与多个 LLM 提供商交互。

---

## 特性

- **多提供商支持** — OpenAI（兼容 Azure）、Anthropic，以及可扩展的 Provider trait
- **自动认证解析** — 从环境变量、凭证存储或自定义上下文自动获取 API Key
- **模型目录管理** — 内置模型元数据（上下文窗口、成本、多模态能力等），支持网络刷新与本地持久化
- **Token 与成本追踪** — 分输入/输出/缓存读写计算用量，自动按阶梯定价计算费用
- **流式响应** — 基于 `async-stream` 的事件流，支持思考块（Thinking）、工具调用（Tool Call）等增量事件
- **工具调用 & 结构化输出** — 通过 `schemars` 生成 JSON Schema，无缝对接 Function Calling
- **上下文持久化** — `Context` 结构支持会话消息累积，可在不同模型间无缝切换
- **重试机制** — 可配置的指数退避重试策略，带抖动
- **线程安全** — 核心类型（`Models` 等）基于 `parking_lot` RwLock 设计，可跨线程安全共享

---

## 安装

在 `Cargo.toml` 中添加：

```toml
[dependencies]
otter-ai = "0.1"
```

默认启用所有内置 Provider（OpenAI、Anthropic、Faux）。如需自定义启用的 Provider，请参见 [功能标志](#功能标志)。

### Rust 版本要求

`otter-ai` 需要 Rust **1.75** 或更高版本。

---

## 功能标志

| 标志 | 默认 | 说明 |
|------|------|------|
| `providers-openai` | ✅ | 启用 OpenAI 兼容提供商（OpenAI、Azure OpenAI 等） |
| `providers-anthropic` | ✅ | 启用 Anthropic 提供商（Claude 系列模型） |
| `providers-faux` | ✅ | 启用 Faux Mock 提供商（用于离线测试） |

只启用 OpenAI 的示例：

```toml
[dependencies]
otter-ai = { version = "0.1", default-features = false, features = ["providers-openai"] }
```

---

## 提供商配置

### 环境变量

SDK 会自动从环境变量读取各提供商的凭证：

| 提供商 | 环境变量 | 说明 |
|--------|----------|------|
| **OpenAI** | `OPENAI_API_KEY` | OpenAI API Key |
| | `OPENAI_BASE_URL` | （可选）自定义 API Base URL，兼容代理 / 中转服务 |
| | `OPENAI_ORG_ID` | （可选）组织 ID |
| **Anthropic** | `ANTHROPIC_API_KEY` | Anthropic API Key |
| | `ANTHROPIC_BASE_URL` | （可选）自定义 API Base URL |
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` | Azure API Key |
| | `AZURE_OPENAI_ENDPOINT` | Azure 端点 URL |

也可以通过 `CredentialStore` trait 实现自定义凭证来源。

---

## 快速开始

```rust
use otter_ai::*;
use otter_ai::types::*;
use otter_ai::models::create_models;
use otter_ai::providers::faux::register_faux_provider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 创建模型注册表
    let models = create_models();

    // 2. 注册一个 Provider（此处用 Faux Mock 作为示例）
    let reg = register_faux_provider(None);
    models.set_provider_arc(reg.provider.clone());

    // 3. 刷新模型目录（allow_network=false, force=true）
    models.refresh_all(false, true).await?;

    // 4. 查找模型
    let model = models.get_model("faux", "faux-mini")
        .expect("model not found");

    // 5. 构建对话上下文
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".into()),
        messages: vec![Message::User {
            content: vec![ContentBlock::Text {
                text: "Hello! 用一句话介绍你自己。".into()
            }],
            timestamp: chrono::Utc::now().timestamp_millis(),
        }],
        ..Default::default()
    };

    // 6. 发起完整补全请求
    let options = SimpleStreamOptions::default();
    let response = models.complete(&model, context, options).await?;

    // 7. 提取并打印文本回复
    if let Message::Assistant { content, usage, .. } = &response {
        println!("回复: {}", content_text(content));
        if let Some(usage) = usage {
            println!("Token 用量: input={}, output={}, total=${:.6}",
                usage.input, usage.output, usage.cost.total);
        }
    }

    Ok(())
}
```

### 流式响应示例

```rust
use futures::StreamExt;

let options = SimpleStreamOptions::default();
let mut stream = models.stream(&model, context, options).await?;

while let Some(event) = stream.next().await {
    match event {
        AssistantMessageEvent::TextDelta { delta, .. } => {
            print!("{}", delta);
        }
        AssistantMessageEvent::Done { message, .. } => {
            println!("\n--- 完成 ---");
        }
        _ => {}
    }
}
```

---

## 核心模块

| 模块 | 说明 |
|------|------|
| [`types`](https://docs.rs/otter-ai/latest/otter_ai/types/index.html) | 核心类型定义：`Model`、`Context`、`Message`、`ContentBlock`、`Usage`、`Tool` 等 |
| [`models`](https://docs.rs/otter-ai/latest/otter_ai/models/index.html) | `Models` 注册表：Provider 管理、模型查询、`complete()` / `stream()` 统一入口 |
| [`providers`](https://docs.rs/otter-ai/latest/otter_ai/providers/index.html) | Provider trait 及内置实现（`openai`、`anthropic`、`faux`） |
| [`auth`](https://docs.rs/otter-ai/latest/otter_ai/auth/index.html) | 认证上下文、凭证存储、自动解析逻辑 |
| [`models_store`](https://docs.rs/otter-ai/latest/otter_ai/models_store/index.html) | 模型目录持久化 trait 及内存实现 |
| [`utils`](https://docs.rs/otter-ai/latest/otter_ai/utils/index.html) | 重试、事件流、JSON 解析、参数校验、成本计算等工具 |

---

## 许可证

本项目采用 **MIT 许可证** 或 **Apache License 2.0** 双重许可，任选其一。

- MIT: [LICENSE-MIT](LICENSE-MIT) 或 <http://opensource.org/licenses/MIT>
- Apache-2.0: [LICENSE-APACHE](LICENSE-APACHE) 或 <http://www.apache.org/licenses/LICENSE-2.0>
