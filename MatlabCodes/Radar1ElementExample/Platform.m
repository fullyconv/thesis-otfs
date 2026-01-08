classdef Platform
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        NAntenna
        Velocity
        Position
        
    end

    methods
        
        function obj = Platform(NAntenna, Velocity, Position)
            if nargin < 3
                error('Not enough input arguments. Please provide NAntenna, Velocity, and Position.');
            end
            obj.NAntenna = NAntenna;
            obj.Velocity = Velocity;
            obj.Position = Position;
        end
        function obj = untitled(inputArg1,inputArg2)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end
    end
end