export const siteConfig = {
  name: "Wenlin Lab",
  title: "Building, Learning, Sharing",
  description:
    "Personal Operating System — 一个集个人品牌、项目展示、学习记录、知识沉淀、公开成长路线图于一体的现代化个人数字实验室。",
  author: "夏阳",
  nav: [
    { label: "首页", href: "/" },
    { label: "项目", href: "/projects" },
    { label: "学习", href: "/learning" },
    { label: "花园", href: "/garden" },
    { label: "关于", href: "/about" },
  ],
  social: {
    github: "https://github.com/LIU3254621731",
    email: "3254621731@qq.com",
    emailGmail: "l18398916038@gmail.com",
  },
};

export const projects = [
  {
    id: "rppg-heart-rate",
    title: "rPPG 远程生理感知系统",
    category: "AI / 计算机视觉",
    description:
      "通过普通摄像头非接触式测量心率和呼吸率，融合元学习、自监督学习和证据深度学习三种前沿范式。",
    tags: ["Python", "PyTorch", "MediaPipe", "MAML", "计算机视觉"],
    featured: true,
    status: "released",
    github_url: "https://github.com",
    demo_url: "",
    image: null,
  },
  {
    id: "vtuber-hub",
    title: "VtuberHub — 全栈虚拟主播套件",
    category: "桌面应用 / 3D",
    description:
      "跨 WPF、Godot、Unity 的完整虚拟主播工具集，基于 MediaPipe 实现实时 3D 角色驱动。",
    tags: ["C#", "Godot", "Unity", "WPF", "MediaPipe", "3D"],
    featured: true,
    status: "building",
    github_url: "https://github.com",
    demo_url: "",
    image: null,
  },
  {
    id: "llm-wiki",
    title: "LLM 智能知识 Wiki",
    category: "全栈 / AI",
    description:
      "基于 Tauri + Rust 构建的 AI 原生知识管理系统，本地优先、隐私安全，支持 LLM 检索与总结。",
    tags: ["Rust", "Tauri", "TypeScript", "LLM", "RAG"],
    featured: true,
    status: "building",
    github_url: "https://github.com",
    demo_url: "",
    image: null,
  },
];
