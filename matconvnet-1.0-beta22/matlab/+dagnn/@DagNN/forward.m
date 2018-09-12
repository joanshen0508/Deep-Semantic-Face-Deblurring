function forward(obj, inputs)

% jasonlai: forward

% obj.computingDerivative = nargin > 2 && ~isempty(derOutputs) ;
obj.computingDerivative = strcmp(obj.mode, 'normal');

if ~iscell(inputs), error('INPUTS is not a cell array.') ; end
% if obj.computingDerivative && ~iscell(derOutputs), error('DEROUTPUTS is not a cell array.') ; end

% -------------------------------------------------------------------------
% Forward pass
% -------------------------------------------------------------------------

% set the input values
v = obj.getVarIndex(inputs(1:2:end)) ;
if any(isnan(v))
    broken = find(isnan(v)) ;
    error('No variable of name ''%s'' could be found in the DAG.', inputs{2*broken(1)-1}) ;
end
[obj.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

obj.numPendingVarRefs = [obj.vars.fanout] ;
for l = obj.executionOrder
    time = tic ;
    %   obj.layers(l)
    
    obj.layers(l).block.forwardAdvanced(obj.layers(l)) ;
    obj.layers(l).forwardTime = toc(time) ;
end