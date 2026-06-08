## Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `id` | `string` | Unique prompt identifier. Zero-padded to 5 digits. |
| 2 | `author_reputation` | `integer` | Author's reputation score on the platform. |
| 3 | `version` | `integer` | How many times the prompt has been revised. |
| 4 | `fork_count` | `integer` | Number of times other users forked (copied & modified) this prompt. |
| 5 | `likes` | `integer` | Total likes received. |
| 6 | `upvotes` | `integer` | Community upvotes. |
| 7 | `downvotes` | `integer` | Community downvotes. |
| 8 | `views` | `integer` | Total view count. |
| 9 | `uses` | `integer` | Number of times the prompt was actually sent to an LLM through the platform. |
| 10 | `created_at` | `string` | ISO 8601 UTC timestamp of when the prompt was first published. Spans roughly the last 2 years. |
| 11 | `title` | `string` | Short title for the prompt. |
| 12 | `content` | `string` | The actual prompt text a user would send to an LLM. This is the **primary semantic field**. |
| 13 | `category` | `string` | Lowercase, hyphenated topic category (e.g. `"coding"`, `"creative-writing"`, `"data-analysis"`, `"marketing"`). Assigned per prompt. |
| 14 | `subcategory` | `string` | More specific label within the category (e.g. `"performance-analysis"`, `"email-marketing"`). Lowercase, hyphenated. |
| 15 | `tags` | `array[string]` | 2–8 lowercase descriptive tags (e.g. `["nodejs", "performance", "debugging"]`). |
| 16 | `has_placeholders` | `boolean` | `true` if the prompt contains `{{variable}}` template placeholders, `false` otherwise. |
| 17 | `placeholders` | `array[string]` | List of placeholder variable names found in `content` (without braces). Empty array `[]` if none. Examples: `["language", "code_snippet"]`. |
| 18 | `difficulty` | `string` | One of: **`beginner`**, **`intermediate`**, **`advanced`**, **`expert`**. |
| 19 | `language` | `string` | ISO 639-1 language code of the prompt content. Predominantly `"en"`, with occasional `"it"`, `"es"`, `"fr"`, `"de"`, `"pt"`, `"zh"`, `"ja"`. |
| 20 | `target_model` | `string` | The LLM the author intended the prompt for. |

---