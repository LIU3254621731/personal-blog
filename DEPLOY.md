# 🚀 EdgeOne 部署指南

## 前置准备

### 你需要有的
- GitHub 仓库：`https://github.com/LIU3254621731/personal-blog`
- EdgeOne Pages 账号（腾讯云）

### 不需要的
- ❌ 域名（EdgeOne 提供测试域名）
- ❌ CloudBase（首次部署用内存模式，页面能渲染）

---

## 第一步：EdgeOne Pages 连接

1. 打开 [EdgeOne 控制台](https://console.cloud.tencent.com/edgeone)
2. 进入 **Pages** → **创建项目**
3. 授权 GitHub → 选择 `personal-blog` 仓库
4. 构建设置：
   ```
   框架预设：  Next.js
   构建命令：  npm run build
   输出目录：  .next
   安装命令：  npm install
   Node 版本： 20
   ```

---

## 第二步：环境变量

在 EdgeOne Pages → 项目设置 → 环境变量中添加：

| 变量名 | 值（测试用） | 说明 |
|--------|-------------|------|
| `ADMIN_PASSWORD_HASH` | 运行 `node scripts/setup-env.js` 获取 | bcrypt 哈希 |
| `JWT_SECRET` | 运行 `node scripts/setup-env.js` 获取 | JWT 签名密钥 |
| `NODE_ENV` | `production` | 生产模式 |

> ⚠️ 密码哈希中的 `$` 符号需要在 EdgeOne 控制台中正常输入（不会被 dotenv 展开，EdgeOne 直接注入环境变量）

---

## 第三步：部署

点击"保存并部署"，EdgeOne 会自动：

1. `npm install`（会跳过 `better-sqlite3` 的 C++ 编译，因为它不在 Edge 环境中）
2. `npm run build`（**最新 db.ts 已做优雅降级**，SQlite 不可用时用内存存贮）
3. 部署到 Edge 节点

---

## 第四步：验证

部署完成后访问 EdgeOne 提供的测试域名（如 `xxx.edgeone.app`）：

| 检查项 | 预期 |
|--------|------|
| 首页 | 渲染正常（Hero + 文章 + 项目 + 热力图） |
| `/blog` | 文章列表正常（SSG 预渲染） |
| `/projects` | 项目列表正常 |
| `/roadmap` | 路线图页面正常 |
| `/resources` | 资源页面正常 |
| `/about` | 关于页面正常 |
| `/settings` | 重定向到首页（未登录） |
| `/feed.xml` | RSS 正常 |
| `/sitemap.xml` | Sitemap 正常 |

### 已知限制（CloudBase 迁移前）

| 功能 | 状态 | 说明 |
|------|------|------|
| 静态页面 | ✅ 完全正常 | SSG 预渲染的页面 |
| 管理员登录 | ⚠️ 部分可用 | JWT 验证正常，但数据不持久 |
| 文章/项目增删改 | ⚠️ 内存存贮 | 部署重启后数据丢失 |
| 图片上传 | ❌ 暂不可用 | 需要 CloudBase Storage |
| 花园/Learning | ⚠️ 部分可用 | MDX 文件在构建时预编译 |

---

## 下一步：接入 CloudBase

等创建好 CloudBase 环境后，只需修改 `src/lib/db.ts`：

1. 把 `dbAvailable` 的检测逻辑改为连接 CloudBase
2. API 路由和页面代码**无需修改**（接口已统一）

---

## 快速命令参考

```bash
# 本地生产构建测试
npm run build

# 生成密码哈希（用于环境变量）
node scripts/setup-env.js

# Git 推送（需先配置 GitHub 凭据）
git push origin master
```
