classdef PrioritizedReplayBuffer < rl.util.ExperienceBuffer
    properties
        PriorityFactor = 0.5; % Factor to prioritize experiences
    end
    
    methods
        function obj = PrioritizedReplayBuffer(capacity, obsInfo, actInfo)
            obj@rl.util.ExperienceBuffer(capacity, obsInfo, actInfo);
        end
        
        function add(obj, experiences, uncertainty)
            % Modify the add method to prioritize experiences
            priority = 1 + obj.PriorityFactor * uncertainty;
            obj.Buffer(obj.NextIndex,:) = {experiences, priority};
            obj.NextIndex = mod(obj.NextIndex, obj.Capacity) + 1;
        end
        
        function [miniBatch, indices] = sample(obj, batchSize)
            % Sampling based on priority
            priorities = cell2mat(obj.Buffer(:, 2));
            probs = priorities / sum(priorities);
            indices = randsample(1:obj.Capacity, batchSize, true, probs);
            miniBatch = obj.Buffer(indices, 1);
        end
    end
end