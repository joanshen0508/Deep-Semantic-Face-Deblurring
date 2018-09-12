function backward(obj,derOutputs)
% -------------------------------------------------------------------------
% Backward pass
% -------------------------------------------------------------------------

% if ~obj.computingDerivative, return ; end
obj.computingDerivative = strcmp(obj.mode, 'normal');

% set output derivatives

v = obj.getVarIndex(derOutputs(1:2:end)) ;
if ~isnan(v)
    [obj.vars(v).der] = deal(derOutputs{2:2:end}) ;
    derOutputs = [] ;
end

obj.numPendingVarRefs = zeros(1, numel(obj.vars)) ;
obj.numPendingParamRefs = zeros(1, numel(obj.params)) ;
for l = fliplr(obj.executionOrder)
    time = tic ;
    %   obj.layers(l)
    obj.layers(l).block.backwardAdvanced(obj.layers(l)) ;
    obj.layers(l).backwardTime = toc(time) ;
end
