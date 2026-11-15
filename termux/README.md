# Подсветка U-Ray в Termux

В Termux подсветка подключается через редактор. В репозитории есть готовые
настройки для **nano**, **Vim/Neovim** и **micro**; файлы `.ur` распознаются
автоматически.

Установи редакторы, если их ещё нет:

```sh
pkg install nano vim neovim micro
```

## nano

Из корня репозитория выполни:

```sh
mkdir -p "$HOME/.config/nano"
cp termux/nano/uray.nanorc "$HOME/.config/nano/uray.nanorc"
touch "$HOME/.nanorc"
include_line="include \"$HOME/.config/nano/uray.nanorc\""
grep -qxF "$include_line" "$HOME/.nanorc" || \
  printf '%s\n' "$include_line" >> "$HOME/.nanorc"
```

После этого открой `.ur` файл командой `nano file.ur`. Команда безопасно
добавляет настройку в `~/.nanorc` только один раз.

## Vim

Для Vim настройка устанавливается так:

```sh
mkdir -p "$HOME/.vim/ftdetect" "$HOME/.vim/syntax"
cp termux/vim/ftdetect/uray.vim "$HOME/.vim/ftdetect/uray.vim"
cp termux/vim/syntax/uray.vim "$HOME/.vim/syntax/uray.vim"
```

Затем открой `.ur` файл через `vim file.ur`.

## micro

Из корня репозитория скопируй syntax-файл:

```sh
mkdir -p "$HOME/.config/micro/syntax"
cp termux/micro/uray.yaml "$HOME/.config/micro/syntax/uray.yaml"
```

Теперь `micro file.ur` автоматически выберет подсветку U-Ray.

### Neovim

Для Neovim используй его каталог конфигурации вместо `~/.vim`:

```sh
mkdir -p "$HOME/.config/nvim/ftdetect" "$HOME/.config/nvim/syntax"
cp termux/vim/ftdetect/uray.vim "$HOME/.config/nvim/ftdetect/uray.vim"
cp termux/vim/syntax/uray.vim "$HOME/.config/nvim/syntax/uray.vim"
```
