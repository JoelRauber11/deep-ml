import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    result = 0
    
    match norm_type:
        case 'l1':
            result = torch.sum(torch.abs(arr))
        case 'l2':
            result = torch.sqrt(torch.sum(arr ** 2))
        case 'linf':
            result = torch.max(torch.abs(arr))
        case 'frobenius':
            if arr.ndim != 2:
                raise ValueError("Wrong length");
            result = torch.sqrt((torch.sum(arr ** 2)))
        case _:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    return float(result)
