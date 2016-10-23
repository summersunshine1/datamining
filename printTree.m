function printTree(tree)
if tree.pro==0
    fprintf('(%d)',tree.value);
    if tree.parentpro~=-1
        fprintf('its parent feature value:%d\n',tree.parentpro);
    end
    return
end
fprintf('[%d]\n',tree.value);
if tree.parentpro~=-1
    fprintf('its parent feature value:%d\n',tree.parentpro);
end
fprintf('its subtree:\n');
childset = tree.child;
for i=1:size(childset,2)
    printTree(childset(i));
end
fprintf('\n');
fprintf('its subtree end\n');
end
    
    