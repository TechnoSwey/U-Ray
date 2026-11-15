if exists("b:current_syntax")
  finish
endif

syntax case match
syntax keyword urayKeyword if else while for in break continue func return import del try catch throw match type
syntax keyword urayBoolean true false
syntax keyword urayBuiltin print
syntax match urayNumber /\<\d\+\(\.\d\+\)\?\>/
syntax region urayString start=/"/ skip=/\\./ end=/"/
syntax match urayComment /\/\/.*$/ contains=urayTodo
syntax keyword urayTodo TODO FIXME XXX contained
syntax match urayFunction /\<[A-Za-z_][A-Za-z0-9_]*\ze\s*(/
syntax match urayModule /\<\(nn\|math\|linalg\)\>/
syntax match urayOperator /==\|!=\|<=\|>=\|[+*\/%=<>-]/

highlight default link urayKeyword Statement
highlight default link urayBoolean Boolean
highlight default link urayBuiltin Function
highlight default link urayNumber Number
highlight default link urayString String
highlight default link urayComment Comment
highlight default link urayTodo Todo
highlight default link urayFunction Function
highlight default link urayModule Include
highlight default link urayOperator Operator

let b:current_syntax = "uray"
