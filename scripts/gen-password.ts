/**
 * Generate a bcrypt hash of a password for use in the ADMIN_PASSWORD_HASH env var.
 *
 * Usage:
 *   npx tsx scripts/gen-password.ts my-admin-password
 *
 * Or interactive:
 *   npx tsx scripts/gen-password.ts
 */

import bcrypt from "bcryptjs";
import { createInterface } from "readline";

async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 12);
}

async function main() {
  const args = process.argv.slice(2);
  let password = args[0];

  if (!password) {
    // Interactive mode
    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    password = await new Promise<string>((resolve) => {
      rl.question("Enter admin password: ", (answer) => {
        rl.close();
        resolve(answer);
      });
    });

    if (!password) {
      console.error("Error: password cannot be empty");
      process.exit(1);
    }
  }

  if (password.length < 8) {
    console.warn("⚠  Warning: password is less than 8 characters — consider a stronger one.\n");
  }

  const hash = await hashPassword(password);

  console.log("\n✅  Password hash generated:\n");
  console.log(`ADMIN_PASSWORD_HASH="${hash}"\n`);
  console.log("Copy the line above into your .env file.\n");
  console.log("Also generate a JWT secret:");
  console.log(`JWT_SECRET="${Array.from(crypto.getRandomValues(new Uint8Array(32)), (b) => b.toString(16).padStart(2, "0")).join("")}"`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
