function status = isSubclass(classname,parentname)
%ISSUBCLASS check if a class inherits from a parent class
%
% Synopsis  status = isSubclass(classname,parentname)
%
% Input:
%   classname - a string giving the name of the class (child)
%
%   parentname - a string giving the name of the parent class
%
% Output
%   status - boolean. Return true if class is a child of parent. False
%   otherwise
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

% Sebastien Besson, Jan 2012
% Generalization of xunit.utils.isTestCaseSubClass from Steven L. Eddins

% Input check
ip =inputParser;
ip.addRequired('classname',@ischar);
ip.addRequired('parentname',@ischar);
ip.parse(classname,parentname);

% Check validity of child class
class_meta = meta.class.fromName(classname);
assert(~isempty(class_meta),' %s is not a valid class',classname);

% Check validity of parent class
assert(~isempty(meta.class.fromName(parentname)),' %s is not a valid parent class',parentname);

% Call recursively class name
status = isMetaSubclass(class_meta,parentname);

function status = isMetaSubclass(class_meta,parentname)

% Initialize output
status = false;

if strcmp(class_meta.Name, parentname), status = true; return; end

% Call function recursively on parent classes (for multiple inheritance)
super_classes = class_meta.SuperClasses;
for k = 1:numel(super_classes)
    if isMetaSubclass(super_classes{k},parentname)
        status = true;
        break;
    end
end

