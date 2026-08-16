# otter-ai

[![crates.io](https://img.shields.io/crates/v/otter-ai.svg)](https://crates.io/crates/otter-ai)
[![docs.rs](https://docs.rs/otter-ai/badge.svg)](https://docs.rs/otter-ai)
[![license](https://img.shields.io/crates/l/otter-ai.svg)](#license)

> 统一的大语言模型 API SDK — 多提供商聚合、自动认证解析、Token 与成本追踪、上下文持久化。

<!-- sync:version:BEGIN --> 📦 0.2.0 · 🦀 Rust ≥ 1.92 · Source: https://github.com/recover965455/otter-ai <!-- sync:version:END -->

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
otter-ai = "0.2"
```

默认启用所有内置 Provider（OpenAI、Anthropic、Faux）。如需自定义启用的 Provider，请参见 [功能标志](#功能标志)。

### Rust 版本要求（MSRV）

`otter-ai` 需要 Rust **1.92** 或更高版本（与 `Cargo.toml` 中 `rust-version` 保持一致；CI 编译使用最新 stable，实际要求只会 ≥ 此处标注）。

---

## 功能标志

`otter-ai` 默认启用所有内置 provider（与 `@earendil-works/pi-ai` 的打包策略一致）。如需只启用少量 provider 以缩短编译时间，请使用 `default-features = false` 再手动开启需要的 feature。

| 标志 | 默认 | 说明 |
|------|------|------|
| `providers-openai` | ✅ | OpenAI（GPT-4o / GPT-4o Mini / o1 / o3-mini，Chat Completions + Responses API） |
| `providers-anthropic` | ✅ | Anthropic（Claude 3 Opus / 3.5 Sonnet / 3.7 Sonnet，Messages API + 缓存定价） |
| `providers-faux` | ✅ | Faux Mock 提供商（用于离线测试、确定性单元测试） |
| `providers-ant-ling` | ✅ | Ant Ling（蚂蚁集团 AI 编程助手） |
| `providers-azure` | ✅ | Azure OpenAI Responses（`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`） |
| `providers-baseten` | ✅ | Baseten 托管推理平台 |
| `providers-bedrock` | ✅ | Amazon Bedrock（ConverseStream 协议 + SigV4 Bearer Token） |
| `providers-cerebras` | ✅ | Cerebras 极速推理（Llama 系列） |
| `providers-chatgpt-plus` | ✅ | ChatGPT Plus/Pro (Codex) 订阅 OAuth |
| `providers-claude-pro-max` | ✅ | Claude Pro/Max 订阅 OAuth |
| `providers-cloudflare-ai-gateway` | ✅ | Cloudflare AI Gateway（聚合代理） |
| `providers-cloudflare-workers-ai` | ✅ | Cloudflare Workers AI（边缘推理，`@cf/*` 模型） |
| `providers-deepseek` | ✅ | DeepSeek V3 Chat + R1 Reasoner |
| `providers-fireworks` | ✅ | Fireworks AI 高速推理 |
| `providers-github-copilot` | ✅ | GitHub Copilot 订阅 OAuth（device-code flow） |
| `providers-google` | ✅ | Google Generative AI（Gemini 2.5 Pro / 2.0 Flash 等） |
| `providers-google-vertex` | ✅ | Google Vertex AI（企业级 Google Cloud LLM） |
| `providers-groq` | ✅ | Groq 极速推理（Llama / Mixtral 系列） |
| `providers-mistral` | ✅ | Mistral / Codestral |
| `providers-moonshot` | ✅ | Moonshot AI / Kimi Coding |
| `providers-nvidia` | ✅ | NVIDIA NIM（`integrate.api.nvidia.com/v1`） |
| `providers-openrouter` | ✅ | OpenRouter 聚合网关（自动注入 `HTTP-Referer` / `X-Title` 头） |
| `providers-openrouter-oauth` | ✅ | OpenRouter OAuth（PKCE，铸造用户可控 API Key） |
| `providers-qwen-token-plan` | ✅ | Qwen Token Plan Individual（千问国际版） |
| `providers-radius` | ✅ | Radius（pi-messages 网关） |
| `providers-vercel-ai-gateway` | ✅ | Vercel AI Gateway（代理 + 监控） |
| `providers-xai` | ✅ | xAI（Grok 3 / 4 系列） |
| `providers-xai-subscription` | ✅ | xAI 订阅（Grok/X subscription，device OAuth） |
| `providers-zai` | ✅ | ZAI / Codin |

只启用 OpenAI 的示例：

```toml
[dependencies]
otter-ai = { version = "0.2", default-features = false, features = ["providers-openai"] }
```

---

## 提供商配置

### 环境变量

SDK 会自动从环境变量读取各提供商的凭证（与 `@earendil-works/pi-ai` 使用完全一致的环境变量名；迁移项目时无需修改）：

| 提供商 | 环境变量 | 说明 |
|--------|----------|------|
| **OpenAI** | `OPENAI_API_KEY` | OpenAI API Key |
| | `OPENAI_BASE_URL` | （可选）自定义 API Base URL，兼容代理 / 中转服务 |
| | `OPENAI_ORG_ID` | （可选）组织 ID |
| **Anthropic** | `ANTHROPIC_API_KEY` | Anthropic API Key |
| | `ANTHROPIC_BASE_URL` | （可选）自定义 API Base URL |
| **Ant Ling** | `ANT_LING_API_KEY` | Ant Ling API Key |
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` | Azure API Key |
| | `AZURE_OPENAI_ENDPOINT` | Azure 端点 URL |
| **Baseten** | `BASETEN_API_KEY` | Baseten API Key |
| **Amazon Bedrock** | `AWS_BEARER_TOKEN_BEDROCK` | 已签名的 Bedrock ConverseStream Bearer Token |
| **Cerebras** | `CEREBRAS_API_KEY` | Cerebras API Key |
| **ChatGPT Plus/Pro (Codex)** | — | 订阅型，通过 `/login` OAuth 授权，无需环境变量 |
| **Claude Pro/Max** | — | 订阅型，通过 `/login` OAuth 授权，无需环境变量 |
| **Cloudflare** | `CLOUDFLARE_API_KEY` | （Cloudflare AI Gateway **和** Workers AI 共用） |
| | `CLOUDFLARE_ACCOUNT_ID` | Cloudflare 账号 ID（嵌入 URL path） |
| | `CLOUDFLARE_GATEWAY_ID` | （仅 AI Gateway）网关实例 ID |
| **DeepSeek** | `DEEPSEEK_API_KEY` | DeepSeek API Key |
| **Fireworks AI** | `FIREWORKS_API_KEY` | Fireworks AI API Key |
| **GitHub Copilot** | — | 订阅型，通过 `/login` device-code OAuth 授权 |
| **Google Gemini** | `GEMINI_API_KEY` | Google Generative AI API Key（aistudio.google.com） |
| **Google Vertex AI** | `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud 服务账号凭证 JSON 路径（或 `gcloud auth print-access-token` 生成的 token） |
| **Groq** | `GROQ_API_KEY` | Groq API Key（以 `gsk_` 开头） |
| **Mistral** | `MISTRAL_API_KEY` | Mistral API Key（以 `trMZ` 开头） |
| **Moonshot AI (Kimi)** | `MOONSHOT_API_KEY` | Moonshot AI API Key |
| **NVIDIA NIM** | `NVIDIA_API_KEY` | NVIDIA NIM API Key（以 `nvapi-` 开头） |
| **OpenRouter** | `OPENROUTER_API_KEY` | OpenRouter API Key |
| **OpenRouter (OAuth)** | — | OAuth 授权铸造 API Key，无需环境变量（`OPENROUTER_API_KEY` 仍可用于 API Key 模式） |
| **Qwen Token Plan** | `QWEN_TOKEN_PLAN_API_KEY` | 千问国际版 API Key |
| **Radius** | — | 订阅型，通过 `/login` OAuth 授权，无需环境变量 |
| **Vercel AI Gateway** | `AI_GATEWAY_API_KEY` | Vercel 颁发的 Key |
| **xAI** | `XAI_API_KEY` | xAI（Grok）API Key |
| **xAI (subscription)** | — | 订阅型，通过 `/login` device-code OAuth 授权（`XAI_API_KEY` 仍可用于 API Key 模式） |
| **ZAI / Codin** | `ZAI_API_KEY` | ZAI API Key |

也可以通过 `CredentialStore` trait 实现自定义凭证来源。

### 配置目录（`~/.otter`）

otter-ai 的本地配置文件统一存放在 **`~/.otter`** 目录下（区别于 TypeScript 版 pi-ai 使用的 `~/.pi`，二者并存互不干扰）。解析顺序：

1. **环境变量覆盖** — `OTTER_CONFIG_DIR=/custom/path` 优先级最高。
2. **XDG Base Directory** — 如果设置了 `XDG_CONFIG_HOME`，使用 `$XDG_CONFIG_HOME/otter`。
3. **默认** — `~/.otter`（Windows 上为 `%USERPROFILE%\.otter`）。

常用的子路径约定（与 pi-ai 保持语义一致，只是换了根目录）：

| 文件 / 目录 | 用途 |
|---|---|
| `~/.otter/agent/auth.json` | `/login` 存储的 API Key / OAuth 凭证 |
| `~/.otter/agent/models-store.json` | 离线模型目录缓存（`refresh_models(allow_network=false)` 时使用） |
| `~/.otter/agent/models.json` | 用户自定义 provider / 模型声明（Ollama、vLLM、代理等） |

SDK 提供三个便捷函数：

```rust,no_run
use otter_ai::{config_dir, config_path, ensure_config_dir};

# async fn example() -> anyhow::Result<()> {
// 获取配置根目录（~/.otter）
let root = config_dir()?.expect("no home directory available");

// 拼接子路径：~/.otter/agent/auth.json
let auth_file = config_path("agent/auth.json")?.expect("no home directory available");

// 创建多级目录（等价于 mkdir -p），返回最终路径
let agent_dir = ensure_config_dir("agent")?.expect("no home directory available");
# Ok(())
# }
```

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
