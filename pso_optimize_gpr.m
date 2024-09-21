function [best_pos, best_val] = pso_optimize_gpr(gprMdl, lb, ub, num_particles, num_iterations)
    dim = length(lb);

    sobolSeq = sobolset(10);

    x_01 = net(sobolSeq, num_particles);

    initial_positions = bsxfun(@plus, lb, bsxfun(@times, x_01, (ub - lb)));

    x = initial_positions;
    v = zeros(num_particles, dim);
    pbest = x;
    pbest_val = inf(num_particles, 1);
    
    for i = 1:num_particles
        [pbest_val(i), ~] = predict(gprMdl, x(i, :));
    end
    
    [gbest_val, gbest_idx] = min(pbest_val);
    gbest = pbest(gbest_idx, :);
    
    w_max = 0.9;
    w_min = 0.4;
    c1 = 2;
    c2 = 2;
    
    for t = 1:num_iterations
        w = w_max - ((w_max - w_min) / num_iterations) * t;
        
        for i = 1:num_particles
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            v(i, :) = w * v(i, :) + c1 * r1 .* (pbest(i, :) - x(i, :)) + c2 * r2 .* (gbest - x(i, :));
            
            x(i, :) = x(i, :) + v(i, :);
            
            x(i, :) = min(max(x(i, :), lb), ub);
            
            [new_val, ~] = predict(gprMdl, x(i, :));
            
            if new_val < pbest_val(i)
                pbest(i, :) = x(i, :);
                pbest_val(i) = new_val;
            end
        end
        
        [min_val, gbest_idx] = min(pbest_val);
        if min_val < gbest_val
            gbest = pbest(gbest_idx, :);
            gbest_val = min_val;
        end
    end
    
    best_pos = gbest;
    best_val = gbest_val;
end
