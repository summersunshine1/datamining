function result=classify(data, tree)
while tree.pro==1
    childset=tree.child;
    v=tree.value;
    for i=1:size(childset,2)
        child = childset(i);
        if child.parentpro==data(v);
            tree=child;
            break;
        end
    end
end
result=tree.value;
end
    