# git-推送方式

若远程仓库中没有创建与本地分支 `sensors-concept2-formulate_plan` 相对应的远程分支，需要执行以下步骤来推送本地分支到远程仓库：

1. **创建本地分支(若已创建忽略此步骤)：**

   在本地代码库中建立自己的分支
   `git checkout -b sensors-concept2-formulate_plan`

2. **创建远程分支**

   首先，在远程仓库中创建一个与本地分支相对应的远程分支。可以使用以下命令：

   ```shell
   git push origin sensors-concept2-formulate_plan:sensors-concept2-formulate_plan
   ```

   这会将本地分支推送到远程仓库，并在远程仓库中创建一个同名的分支 `sensors-concept2-formulate_plan`。

3. **推送本地分支：**

   一旦远程分支创建成功，可以使用正常的 `git push` 命令来将本地分支推送到远程分支：

   ```shell
   git push origin sensors-concept2-formulate_plan
   ```

   这会将本地分支的修改推送到远程分支中。

4. **将修改添加到暂存区：**

   使用 `git add` 命令将需要要提交的修改添加到暂存区。可以执行以下命令，将所有修改添加到暂存区：

   ```shell
   git add .
   ```

   或者，如果只将特定文件的修改添加到暂存区，可以使用以下命令，将文件名替换为实际的文件名：

   ```shell
   git add include/pm5getdata.h include/sed_data_sql.h include/sed_formulate_plan.h source/main.cpp source/sed_data_sql.cpp
   ```

5. **提交修改到本地分支：**

   使用 `git commit` 命令将暂存区中的修改提交到本地分支。在提交时，需要提供一个有意义的提交信息，描述所做的修改。例如：

   ```shell
   git commit -m "增加Get请求"
   ```

   请将 `"增加Get请求"` 替换为适合本次修改的实际描述。

6. **查看本地分支状态：**

   使用 `git status` 命令来确认提交是否成功，以及是否还有未提交的修改。

   ```shell
   git status
   ```

7. **推送前先pull**

   为避免发生冲突，先`pull`一遍，如有冲突先在本地将冲突解决

   ```shell
   git pull
   ```

8. **推送本地分支到远程仓库**

   将本地分支的修改推送到远程仓库，可以使用 `git push` 命令：

   ```shell
   git push origin sensors-concept2-formulate_plan
   ```

   这会将本地分支的修改推送到远程仓库中的相应分支。

请注意，本地分支可能已经比远程仓库的 `origin/master` 分支超前了 n 次提交。在推送之前，确保本地分支和远程分支保持同步。

