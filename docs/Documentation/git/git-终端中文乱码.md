在 Git 中出现中文文件名乱码的情况，通常是因为终端或 Git 本身的编码设置问题。可以通过以下步骤来解决 Git 显示中文文件名乱码的问题：

### 1. **检查终端编码**
   - 确保你的终端使用的是 UTF-8 编码。可以通过以下命令检查终端的当前编码：
     ```bash
     locale
     ```
   - 确保输出的 `LANG` 和 `LC_CTYPE` 等变量中包含 `UTF-8`，例如：
     ```
     LANG="en_US.UTF-8"
     LC_CTYPE="en_US.UTF-8"
     ```
   - 如果没有设置为 UTF-8，可以在终端中运行以下命令来设置：
     ```bash
     export LC_ALL=en_US.UTF-8
     export LANG=en_US.UTF-8
     ```

### 2. **设置 Git 的编码**
   - 你可以通过设置 Git 配置来确保 Git 以 UTF-8 编码处理文件名：
     ```bash
     git config --global core.quotepath false
     ```
   - 这条命令的作用是禁用 Git 对非 ASCII 字符的转义，确保中文文件名可以正确显示。

### 3. **在 `.gitconfig` 文件中手动配置编码**
   - 打开 Git 的全局配置文件 `.gitconfig`，手动添加或修改编码设置：
     ```bash
     [i18n]
         commitencoding = utf-8
     [core]
         quotepath = false
     ```

完成这些步骤后，再次运行 `git status`，中文文件名应该会正确显示。