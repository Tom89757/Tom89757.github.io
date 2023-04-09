" I like using H and L for beginning/end of line
" nmap H ^
" nmap L $
" "map j j to Esc"
" " imap jj <Esc>
" " Have j and k navigate visual lines rather than logical ones
" nmap j gj
" nmap k gk
" basic settings
nnoremap H ^
nnoremap L $
nnoremap yH y^
nnoremap yL y$
nnoremap j gj
nnoremap k gk

" Yank to system clipboard"
set clipboard=unnamed

set number
set relativenumber

"set some Ctrl- shortcuts"
" Go back and forward with Ctrl+O and Ctrl+I
" (make sure to remove default Obsidian shortcuts for these to work)
" exmap back obcommand app:go-back
" nmap <C-o> :back
" exmap forward obcommand app:go-forward
" nmap <C-i> :forward

" nmap <C-[> <nop>
" nmap <C-]> <nop>

" imap <C-[> <nop>
" imap <C-]> <nop>
" nmap <C-[> <C-d>
" imap <C-[> <C-d>

