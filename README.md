# 寄存器线性化审计服务（Register Linearizability Audit Service）

隔离材料实验室校准寄存器的并发访问审计后端。审计员提交一段**已完成**的操作历史（初始整数值 + 1~24 个操作），服务判定该历史是否**可线性化**：即是否存在一个全序，同时满足

1. **实时前驱约束**：若操作 A 的响应时刻 ≤ 操作 B 的调用时刻，则 A 排在 B 之前；
2. **单副本寄存器语义**：写入总是生效；读取必须返回寄存器当前值；compare-and-swap 的成功标志必须与当前值一致（当前值等于期望值则必须成功并写入更新值，否则必须失败且值不变）。

纯后端实现（Python 3.13 + FastAPI），不含前端，不调用任何在线服务。

## 快速启动

```bash
# 默认宿主机端口 8000
docker compose up --build

# 通过环境变量配置宿主机端口（也支持 .env 文件）
API_PORT=9000 docker compose up --build
```

健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

交互式 API 文档（Swagger UI）：`http://localhost:8000/docs`

## API

### `POST /api/v1/linearizability/check`

请求体：

```json
{
  "initial_value": 0,
  "operations": [
    {"id": "w1", "type": "write", "value": 1, "invoke": 0, "respond": 6},
    {"id": "r1", "type": "read", "value": 0, "invoke": 1, "respond": 2},
    {"id": "c1", "type": "cas", "expected": 1, "update": 2, "success": true, "invoke": 3, "respond": 8}
  ]
}
```

字段规则（任一违反则**整体拒绝**，HTTP 422）：

| 字段 | 规则 |
| --- | --- |
| `initial_value` | 整数（严格类型，不接受字符串/布尔/浮点） |
| `operations` | 1 至 24 个操作 |
| `id` | 非空字符串，各操作唯一 |
| `type` | `write` / `read` / `cas`（亦接受别名 `compare-and-swap`） |
| `invoke` / `respond` | 整数时刻，且必须满足 `invoke <= respond`（非法区间整体拒绝） |
| `write` / `read` | 必须携带 `value`，且不得携带 `expected`/`update`/`success` |
| `cas` | 必须携带 `expected`、`update`、`success`（严格布尔），且不得携带 `value` |
| 其他 | 请求体与操作均不接受未定义的多余字段 |

可线性化时的响应（HTTP 200）：

```json
{
  "linearizable": true,
  "unique": true,
  "order": ["r1", "w1", "c1"],
  "steps": [
    {"id": "r1", "before": 0, "after": 0},
    {"id": "w1", "before": 0, "after": 1},
    {"id": "c1", "before": 1, "after": 2}
  ],
  "message": "linearizable: ..."
}
```

- `order`：按操作标识字典序裁决的**最小**合法全序；
- `steps`：按该全序逐步执行时，每步之前/之后的寄存器值；
- `unique`：合法全序是否唯一（`true` 表示恰有一个，`false` 表示存在多个）。

不可线性化时的响应（HTTP 200，明确无解，不伪造任何部分顺序）：

```json
{
  "linearizable": false,
  "unique": null,
  "order": null,
  "steps": null,
  "message": "not linearizable: no total order respects both ..."
}
```

## 判定算法

`app/solver.py` 与 Web 层完全解耦，可独立测试：

1. 由实时区间构建前驱关系（`respond(A) <= invoke(B)` ⇒ `A ≺ B`），位掩码 Floyd-Warshall 求传递闭包；若成环（如两个同时刻的瞬时操作互相前驱）直接判无解；
2. 深度优先搜索拓扑序空间：每步只在“前驱已全部就位”的操作中尝试候选，并立即校验寄存器语义；候选按标识字典序展开，因此**首个**完整解即为字典序最小全序；
3. 寄存器可能值域压缩（初始值 ∪ 写入值 ∪ CAS 更新值）+ 不可行状态记忆化，24 个操作的对抗性实例亦在秒级内完成；
4. 找到 2 个不同解即停止，据此判定唯一性；无解时不返回任何部分顺序。

## 本地开发与测试

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/uvicorn app.main:app --reload
```

## 项目结构

```
app/
  main.py    # FastAPI 应用：/health 与 /api/v1/linearizability/check
  models.py  # 严格请求校验（区间、唯一标识、类型字段一致性）
  solver.py  # 线性化判定引擎（纯 Python，无外部依赖）
tests/       # 求解器语义测试 + API 校验测试（40 例）
Dockerfile   # python:3.13-slim，非 root 运行，内置 HEALTHCHECK
docker-compose.yml  # API_PORT 环境变量配置宿主机端口
```
