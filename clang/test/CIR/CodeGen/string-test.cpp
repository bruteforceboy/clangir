// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fno-PIE -S -Xclang -emit-cir %s -o %t1.cir
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fno-PIE -S -Xclang -emit-llvm %s -o %t1.ll

#include <string>

void foo() {
  std::string a;
  std::string b;
  a = b;
}
