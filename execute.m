
initializeRobotParameters;

mdl = "rlQuadrupedRobot";
open_system(mdl)
numObs = 44;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "observations";
numAct = 8;
actInfo = rlNumericSpec([numAct 1],LowerLimit=-1,UpperLimit=1);
actInfo.Name = "torque";
blk = mdl + "/RL Agent";
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);
env.ResetFcn = @quadrupedResetFcn;
rng(0)
agentOptions = rlDDPGAgentOptions();
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 256;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.MaxMiniBatchPerEpoch = 200;
agentOptions.NoiseOptions.StandardDeviation = 0.1;
agentOptions.NoiseOptions.MeanAttractionConstant = 1.0; 
agentOptions.ActorOptimizerOptions.Algorithm = "adam";
agentOptions.ActorOptimizerOptions.LearnRate = 1e-3;
agentOptions.ActorOptimizerOptions.GradientThreshold = 1;

agentOptions.CriticOptimizerOptions.Algorithm = "adam";
agentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agentOptions.CriticOptimizerOptions.GradientThreshold = 1;

% Initialize the custom prioritized replay buffer
% experienceBuffer = PrioritizedReplayBuffer(agentOptions.ExperienceBufferLength, obsInfo, actInfo);

% Modify the agent to use the custom buffer
% agentOptions.ExperienceBuffe  r = experienceBuffer;

initOpts = rlAgentInitializationOptions(NumHiddenUnit=256);
agent = rlDDPGAgent(obsInfo,actInfo,initOpts,agentOptions);


% 
agent.ExperienceBuffer = rlPrioritizedReplayMemory(obsInfo,actInfo);
resize(agent.ExperienceBuffer,5e5);
agent.ExperienceBuffer.NumAnnealingSteps = 5e8;
agent.ExperienceBuffer.PriorityExponent = 0.4;
agent.ExperienceBuffer.InitialImportanceSamplingExponent = 0.1;


% Training loop with active learning
% for episode = 1:trainOpts.MaxEpisodes
%     % Reset the environment
%     initialObservation = reset(env);
% 
%     for step = 1:trainOpts.MaxStepsPerEpisode
%         % Get action from agent
%         action = getAction(agent, initialObservation);
% 
%         % Step the environment
%         [nextObservation, reward, isDone] = step(env, action);
% 
%         % Get the Q-values from the critic
%         qValues = getValue(agent.Critic, {initialObservation, action});
% 
%         % Estimate uncertainty as the variance of Q-values
%         uncertainty = var(qValues);
% 
%         % Store experience with uncertainty-based prioritization
%         experiences = {initialObservation, action, reward, nextObservation, isDone};
%         experienceBuffer.add(experiences, uncertainty);
% 
%         % Train the agent with prioritized experience sampling
%         train(agent);
% 
%         % Update observation
%         initialObservation = nextObservation;
% 
%         if isDone
%             break;
%         end
%     end
% 
%     % Periodically evaluate the policy
%     if mod(episode, evaluator.EvaluationFrequency) == 0
%         evaluationScore = evaluatePolicy(agent, env, evaluator.NumEpisodes);
%         fprintf("Episode %d, Evaluation Score: %.2f\n", episode, evaluationScore);
% 
%         if evaluationScore >= trainOpts.StopTrainingValue
%             fprintf("Stopping training at episode %d due to reaching evaluation target.\n", episode);
%             break;
%         end
%     end
% end





trainOpts = rlTrainingOptions(...
    MaxEpisodes=5000,...
    MaxStepsPerEpisode=floor(Tf/Ts),...
    ScoreAveragingWindowLength=250,...
    Plots="training-progress",...
    StopTrainingCriteria="EvaluationStatistic",...
    StopTrainingValue=300, ...
    Verbose=1);

trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = "async";
parpool(4);
doTraining = true;
if doTraining    
    % Evaluate the greedy policy performance by taking the cumulative
    % reward mean over 5 simulations every 25 training episodes.
    evaluator = rlEvaluator(NumEpisodes=5,EvaluationFrequency=25);
    % Train the agent.
    trainingStats = train(agent,env,trainOpts,Evaluator=evaluator);
    save("trainingStats");
else
    % Load pretrained agent parameters for the example.
    load("rlQuadrupedAgentParams.mat","params")
    setLearnableParameters(agent, params);
end
delete(gcp('nocreate'));