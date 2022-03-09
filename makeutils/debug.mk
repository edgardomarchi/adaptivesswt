debug_trace = 1

# $(debug-enter)
debug-enter = $(if $(debug_trace),\
                $(info Entering $0($(echo-args))))

# $(debug-leave)
debug-leave = $(if $(debug_trace),$(info Leaving $0))

comma := ,

echo-args = $(subst ' ','$(comma) ',\
              $(foreach a,1 2 3 4 5 6 7 8 9,'$($a)'))
