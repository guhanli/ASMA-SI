num_particles = 30;
dim = 10; 

w_max = 0.9;
w_min = 0.4;
c1 = 2;
c2 = 2;
num_iterations = 100;

v = zeros(num_particles, dim);

lb = [0.01, 0.01, 0.01, 0.01, 0.01, 0.7, 0.7, 0.7, 0.7, 0.7];
ub = [0.025, 0.025, 0.025, 0.025, 0.025, 1.0, 1.0, 1.0, 1.0, 1.0];

sobolSeq = sobolset(10);

x_01 = net(sobolSeq, num_particles);

x = bsxfun(@plus, lb, bsxfun(@times, x_01, (ub - lb)));

pbest = x;
pbest_val = zeros(num_particles, 1);

samples = [];
targets = [];


for i = 1:num_particles
    pbest_val(i) = objective_function(x(i, :));
    samples = [samples; x(i, :)];
    targets = [targets; pbest_val(i)];
end

[gbest_val, gbest_idx] = min(pbest_val);
gbest = pbest(gbest_idx, :);


gprMdl = fitrgp(samples, targets, ...
    'KernelFunction', 'squaredexponential', 'Standardize', 1);

[proxy_best_pos, proxy_best_val] = pso_optimize_gpr(gprMdl, lb, ub, 30, 100);

real_val = objective_function(proxy_best_pos);

if real_val < gbest_val
    gbest = proxy_best_pos;
    gbest_val = real_val;
end

samples = [samples; proxy_best_pos];
targets = [targets; real_val];

for t = 1:num_iterations
    history_pso_pos = [history_pso_pos; history_pso_pos(end, :)];
    history_pso_val = [history_pso_val; history_pso_val(end)];

    w = w_max - ((w_max - w_min) / num_iterations) * t;
    
    for i = 1:num_particles
        r1 = rand(1, dim);
        r2 = rand(1, dim);
        v(i, :) = w * v(i, :) + c1 * r1 .* (pbest(i, :) - x(i, :)) + c2 * r2 .* (gbest - x(i, :));
        
        x(i, :) = x(i, :) + v(i, :);
        
        x(i, :) = min(max(x(i, :), lb), ub);
        
        new_val = objective_function(x(i, :));
        if new_val < history_pso_val(end)
            history_pso_pos(end, :) = x(i, :);
            history_pso_val(end) = new_val;
        end
        
        if new_val < pbest_val(i)
            pbest(i, :) = x(i, :);
            pbest_val(i) = new_val;
        end

        samples = [samples; x(i, :)];
        targets = [targets; new_val];
    end
    
    [min_val, gbest_idx] = min(pbest_val);
    if min_val < gbest_val
        gbest = pbest(gbest_idx, :);
        gbest_val = min_val;
    end

    gprMdl = fitrgp(samples, targets, ...
        'KernelFunction', 'squaredexponential', 'Standardize', 1);
    
    [proxy_best_pos, proxy_best_val] = pso_optimize_gpr(gprMdl, lb, ub, 30, 100);

    real_val = objective_function(proxy_best_pos);
    
    % 将代理模型找到的最优位置作为候选
    if real_val < gbest_val
        gbest = proxy_best_pos;
        gbest_val = real_val;
    end
    
    samples = [samples; proxy_best_pos];
    targets = [targets; real_val];
end