const bcrypt = require("bcryptjs");
const fs = require("fs");
const crypto = require("crypto");

async function main() {
  const hash = await bcrypt.hash("076215", 12);
  const secret = Array.from(crypto.randomBytes(32), (b) =>
    b.toString(16).padStart(2, "0")
  ).join("");

  // Write to JSON config (no dotenv-expand issues)
  const config = { adminPasswordHash: hash, jwtSecret: secret };
  fs.writeFileSync("auth.config.json", JSON.stringify(config, null, 2), "utf8");
  console.log("auth.config.json written successfully");
  console.log("Hash:", hash);

  // Also write .env with JWT_SECRET only
  const envContent = `# JWT secret for admin authentication\nJWT_SECRET="${secret}"\n`;
  fs.writeFileSync(".env", envContent, "utf8");
  console.log(".env written (JWT_SECRET only)");
}
main();
