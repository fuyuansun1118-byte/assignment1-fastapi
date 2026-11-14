"""Execute every autograd example in Assignment 4 Questions 6-9."""
import json
import torch


def run_gradient_checks():
    result={}
    x=torch.tensor([2.0],requires_grad=True)
    (x**2+3*x).backward()
    assert x.grad.item()==7
    result['Q6a_x_grad']=x.grad.tolist()
    x=torch.tensor([2.0],requires_grad=False)
    try:
        (x**2+3*x).backward()
        raise AssertionError('Expected RuntimeError')
    except RuntimeError as e:result['Q6b_error']=str(e)
    result['Q6c_default_requires_grad']=torch.tensor([2.0]).requires_grad
    x=torch.tensor([2.0],requires_grad=True)
    w=torch.tensor([1.,3.])
    (w[0]*x**2+w[1]*x).backward()
    assert x.grad.item()==7 and w.grad is None
    result['Q7a']={'x_grad':x.grad.tolist(),'w_grad':w.grad}
    x=torch.tensor([2.0]);w=torch.tensor([1.,3.],requires_grad=True)
    (w[0]*x**2+w[1]*x).backward()
    assert w.grad.tolist()==[4.,2.]
    result['Q7b_w_grad']=w.grad.tolist()
    result['Q7c_default_requires_grad']=torch.tensor([1.,3.]).requires_grad
    x=torch.tensor([1.0],requires_grad=True)
    y=x*3;z=y.detach()
    try:
        (z*2).backward()
        raise AssertionError('Expected RuntimeError')
    except RuntimeError as e:result['Q8_original_error']=str(e)
    x=torch.tensor([1.0],requires_grad=True)
    y=x*3;z=y.clone();(z*2).backward()
    assert x.grad.item()==6
    result['Q8_fixed_x_grad']=x.grad.tolist()
    x=torch.tensor([1.0],requires_grad=True)
    (x*2).backward();first=x.grad.item()
    (x*3).backward();second=x.grad.item()
    x.grad=None;(x*3).backward();reset=x.grad.item()
    assert [first,second,reset]==[2.,5.,3.]
    result['Q9']={'first':first,'accumulated_second':second,'after_reset':reset}
    return result

if __name__=='__main__':print(json.dumps(run_gradient_checks(),indent=2))
