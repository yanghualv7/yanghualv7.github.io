# git-小细节

## 项目克隆失败

在开启加速器克隆**GitHub**项目时会存在克隆失败问题

```bash
git clone https://github.com/yanghualv7/yanghualv7.github.io.git pages
```



这个错误提示显示你的电脑无法连接到 GitHub 的服务器

```bash
Cloning into 'pages'...
fatal: unable to access 'https://github.com/yanghualv7/yanghualv7.github.io.git/
': Failed to connect to github.com port 443 after 21127 ms: Couldn't connect to
server
```



此时可以使用下面的方法解决

```bash
git config --global http.postBuffer 524288000
```

这个命令的意义是设置 Git 的 `http.postBuffer` 参数的值为 524288000。

`http.postBuffer` 参数用于设置 Git 在进行 HTTP 网络请求时使用的缓冲区大小。默认情况下，Git 使用较小的缓冲区来处理 HTTP 请求。但是，当你在进行大型操作（如克隆大型存储库）时，上传的数据可能会超出默认缓冲区大小，导致请求失败。

通过执行上述命令并设置较大的 `http.postBuffer` 值，可以增加 Git 的缓冲区大小，以便处理大量数据的上传。这样可以避免在上传大型操作时出现请求失败的情况。

需要注意的是，这个命令是全局设置，对当前用户的所有 Git 仓库都生效。如果你只想在特定的仓库中设置缓冲区大小，可以在该仓库目录下执行相同的命令，而不使用 `--global` 参数。



成功克隆

```bash
git clone https://github.com/yanghualv7/yanghualv7.github.io.git pages
Cloning into 'pages'...
remote: Enumerating objects: 355, done.
remote: Counting objects: 100% (103/103), done.
remote: Compressing objects: 100% (63/63), done.
remote: Total 355 (delta 35), reused 83 (delta 25), pack-reused 252Receiving obj
Receiving objects: 100% (355/355), 12.55 MiB | 4.45 MiB/s, done.

Resolving deltas: 100% (107/107), done.
```
