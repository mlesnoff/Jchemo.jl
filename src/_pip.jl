
struct Pipeline
    mod::Tuple
end

pip(args...) = Pipeline(values(args))

