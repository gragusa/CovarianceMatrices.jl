# claude.md — Write idiomatic, fast Julia

## Core mindset

* Prefer **small, generic functions** + **multiple dispatch** over big if/else trees. Design APIs around *methods on types*, not flags.
* Treat **exported methods as the interface**; avoid poking at struct fields from outside a module.

## Performance first principles

* Put hot code **inside functions**; avoid work in global scope. If you must keep globals, make them `const` or give them a concrete type.
* **Be type-stable**: the result type should be determined by the argument types. Don’t change a variable’s type inside a function.
* Avoid **abstractly-typed containers** (e.g., `Vector{Number}`); use concrete element types or parametric structs - Use parameter in struct when necessary or convenient. 
* **Measure correctly**: use `BenchmarkTools.@btime/@benchmark`, not just `@time`, and interpolate inputs.

## Idioms that unlock speed

* **Broadcast with dots** (`f.(xs)`, `x .+ y`) instead of writing explicit loops *when you want elementwise semantics*. Dots fuse: `y .= 2 .* x .+ 1` creates no temporaries.
* Prefer **preallocation** + mutating forms (`mul!`, `copyto!`, `y .= …`) in tight loops. (Generalization of avoiding temporaries; validated by manual’s allocation advice.)
* Use **views** for slices (`views(A,: , i)`) and **bounds-check control** in hot loops (`@inbounds` only after testing).

## Do / Don’t cheat sheet

* Define small methods; dispatch on types, not values.
* Pass data as arguments; avoid reading/writing non-`const` globals.
* Give arrays concrete eltypes (e.g., `Vector{Float64}`) and structs concrete field types.
* ❌ Don’t rely on vectorization for speed like NumPy; **plain Julia loops are fast** when type-stable.
* ❌ Don’t return different concrete types from different branches of the *same* method. Make result types consistent or split methods.


## Naming & style (brief)

* Use **lowercase\_with\_underscores** for variables/functions when clearer; keep API names consistent. Follow a "sciml" project and code  style.
* Prefer verbs for functions, nouns for types; keep modules focused and export the public surface.
