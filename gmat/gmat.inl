class gmat
{
private:
	float * _data;
	gmat_frame _frame;
public:
	inline gmat() {}
	inline gmat(size_t height, size_t width, int device = 0)
	{
		auto exc = gmat_create(this, height, width, device);
		CHECK_EXC(exc);
	}
	inline gmat(size_t height, size_t width, const void * src, int device = 0)
	{
		auto exc = gmat_create(this, height, width, device);
		CHECK_EXC(exc);
		exc = gmat_copy_from(this, src);
		CHECK_EXC(exc);
	}
	template<typename T>
	inline gmat(size_t height, size_t width, T begin, T end, int device = 0)
	{
		auto exc = gmat_create(this, height, width, device);
		CHECK_EXC(exc);

		float * src = new float[height * width];
		size_t p = 0;
		for (T it = begin; it != end; it++, p++)
			src[p] = (float)(*it);

		for (; p < height * width; p++)
			src[p] = 0.0f;

		exc = gmat_copy_from(this, src);
		delete[] src;
		CHECK_EXC(exc);
	}
	template<typename T>
	inline gmat(const std::initializer_list<std::initializer_list<T>> & init, int device = 0)
	{
		size_t height = init.size();
		size_t width = 0;
		for (auto & i : init)
			if (width < i.size())
				width = i.size();

		auto exc = gmat_create(this, height, width, device);
		CHECK_EXC(exc);

		float * src = new float[height * width];

		size_t hp = 0;
		size_t wp = 0;

		for (auto & i : init)
		{
			wp = 0;
			for (auto & j : i)
				src[hp * width + wp++] = (float)j;
			for (; wp < width; wp++)
				src[hp * width + wp] = 0.0f;
			hp++;
		}

		exc = gmat_copy_from(this, src);
		delete[] src;
		CHECK_EXC(exc);
	}
	inline gmat(size_t size, int device = 0)
	{
		auto exc = gmat_create_identity(this, size, device);
		CHECK_EXC(exc);
	}
	inline gmat(const gmat_frame & frame)
	{
		auto exc = gmat_create(this, frame.height, frame.width, frame.device);
		CHECK_EXC(exc);
	}
	inline gmat(const gmat & o)
	{
		auto exc = gmat_create(this, o._frame.height, o._frame.width, o._frame.device);
		CHECK_EXC(exc);
		exc = gmat_copy(this, &o);
		CHECK_EXC(exc);
	}
	inline gmat(gmat && o)
		: _data(o._data), _frame(o._frame)
	{
		o._data = nullptr;
		o._frame.height = 0;
		o._frame.width = 0;
		o._frame.device = 0;
	}

	inline gmat & operator= (const gmat & o)
	{
		gmat_destroy(this);
		auto exc = gmat_create(this, o._frame.height, o._frame.width, o._frame.device);
		CHECK_EXC(exc);
		exc = gmat_copy(this, &o);
		CHECK_EXC(exc);
	}

	inline gmat & operator= (gmat && o)
	{
		gmat_destroy(this);
		_data = o._data;
		_frame = o._frame;
		o._data = nullptr;
		o._frame.height = 0;
		o._frame.width = 0;
		o._frame.device = 0;
		return *this;
	}

	inline ~gmat()
	{
		gmat_destroy(this);
	}

public:
	inline size_t get_size() const noexcept
	{
		size_t rtn;
		gmat_get_size(this, &rtn);
		return rtn;
	}

	inline size_t get_height() const noexcept
	{
		return _frame.height;
	}

	inline size_t get_width() const noexcept
	{
		return _frame.width;
	}

	inline auto & copy_to(void * dst) const
	{
		auto exc = gmat_copy_to(this, dst);
		CHECK_EXC(exc);
		return *this;
	}

	inline gmat & copy_from(const void * src)
	{
		auto exc = gmat_copy_from(this, src);
		CHECK_EXC(exc);
		return *this;
	}

	inline gmat copy()
	{
		gmat rtn(_frame.height, _frame.width, _frame.device);
		auto exc = gmat_copy(&rtn, this);
		CHECK_EXC(exc);

		return rtn;
	}

	inline gmat copy(int device)
	{
		gmat rtn(_frame.height, _frame.width, device);
		auto exc = gmat_copy(&rtn, this);
		CHECK_EXC(exc);

		return rtn;
	}

	inline gmat transpose()
	{
		gmat rtn(_frame.width, _frame.height, _frame.device);
		auto exc = gmat_transpose(&rtn, this);
		CHECK_EXC(exc);

		return rtn;
	}

	friend gmat operator+ (const gmat &, const gmat &);
	friend gmat operator+ (const gmat &, float);
	friend gmat operator+ (float, const gmat &);
	friend gmat operator- (const gmat &, const gmat &);
	friend gmat operator- (const gmat &, float);
	friend gmat operator- (float, const gmat &);
	friend gmat operator* (const gmat &, const gmat &);
	friend gmat operator* (const gmat &, float);
	friend gmat operator* (float, const gmat &);
	friend gmat operator/ (const gmat &, const gmat &);
	friend gmat operator/ (const gmat &, float);
	friend gmat operator/ (float, const gmat &);
	friend gmat operator% (const gmat &, const gmat &);
	friend gmat operator~ (const gmat &);
	template<typename T>
	friend std::basic_ostream<T> & operator<< (std::basic_ostream<T> &, const gmat &);
};

inline gmat operator+ (const gmat & mat0, const gmat & mat1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_add(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator+ (const gmat & mat0, float val1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_add_float(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator+ (float val0, const gmat & mat1)
{
	gmat rtn(mat1._frame);
	auto exc = gmat_add_float(&rtn, &mat1, val0);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator- (const gmat & mat0, const gmat & mat1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_sub(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator- (const gmat & mat0, float val1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_sub_float(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator- (float val0, const gmat & mat1)
{
	gmat rtn(mat1._frame);
	auto exc = gmat_sub_float_r(&rtn, val0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator* (const gmat & mat0, const gmat & mat1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_mul(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator* (const gmat & mat0, float val1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_mul_float(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator* (float val0, const gmat & mat1)
{
	gmat rtn(mat1._frame);
	auto exc = gmat_mul_float(&rtn, &mat1, val0);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator/ (const gmat & mat0, const gmat & mat1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_div(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator/ (const gmat & mat0, float val1)
{
	gmat rtn(mat0._frame);
	auto exc = gmat_div_float(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator/ (float val0, const gmat & mat1)
{
	gmat rtn(mat1._frame);
	auto exc = gmat_div_float_r(&rtn, val0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator% (const gmat & mat0, const gmat & mat1)
{
	gmat rtn(mat0._frame.height, mat1._frame.width, mat0._frame.device);
	auto exc = gmat_product(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat operator~ (const gmat & mat0)
{
	gmat rtn(mat0._frame.width, mat0._frame.height, mat0._frame.device);
	auto exc = gmat_transpose(&rtn, &mat0);
	CHECK_EXC(exc);
	return rtn;
}

inline gmat & operator+= (gmat & mat0, const gmat & mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmat & operator+= (gmat & mat0, float mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmat & operator-= (gmat & mat0, const gmat & mat1)
{
	return mat0 = mat0 - mat1;
}

inline gmat & operator-= (gmat & mat0, float mat1)
{
	return mat0 = mat0 - mat1;
}

inline gmat & operator*= (gmat & mat0, const gmat & mat1)
{
	return mat0 = mat0 * mat1;
}

inline gmat & operator*= (gmat & mat0, float mat1)
{
	return mat0 = mat0 * mat1;
}

inline gmat & operator/= (gmat & mat0, const gmat & mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmat & operator/= (gmat & mat0, float mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmat & operator%= (gmat & mat0, const gmat & mat1)
{
	return mat0 = mat0 % mat1;
}

inline gmat operator- (const gmat & mat0)
{
	return 0 - mat0;
}

inline gmat operator+ (const gmat & mat0)
{
	return mat0;
}

template<typename T>
inline std::basic_ostream<T> & operator<<(std::basic_ostream<T> & o, const gmat & g)
{
	auto height = g._frame.height;
	auto width = g._frame.width;

	float * tmp = new float[height * width];
	g.copy_to(tmp);

	o << '{';
	for (int i = 0; i < height; i++)
	{
		o << "{ ";
		for (int j = 0; j < width; j++)
		{
			o << tmp[i * width + j] << ", ";
		}
		o << '}';
		if (i != height - 1)
			o << ',' << std::endl;
	}
	o << '}';
	return o;
}
